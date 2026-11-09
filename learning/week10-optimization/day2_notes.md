# Day 2: 分布式 Profiling + 瓶颈→决策映射框架

## 1. 分布式 Profiling 代码路径

### 调用链

```
API /start_profile
  → async_llm.start_profile()                    # entrypoint
    → engine_core.profile_async()                 # 发到 EngineCore
      → core.profile()                            # core.py:769
        → model_executor.profile(is_start=True)   # abstract.py:256
          → collective_rpc("profile", args=(...)) # 广播到所有 worker
```

### collective_rpc 广播机制（multiproc_executor.py:354）

- executor 通过 `rpc_broadcast_mq`（MessageQueue）把方法名 + 参数广播给所有 worker 进程
- 每个 worker 收到消息后，独立在自己的 GPU 上执行 `worker.profile()`
- 响应通过 `response_mqs` 收集，每个 rank 一个队列

### Worker 端执行（gpu_worker.py:1112）

```python
def profile(self, is_start=True, profile_prefix=None):
    # 1. 生成 rank 标识：dp0_pp0_tp{rank}_dcp0_ep0_rank{global_rank}
    rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

    # 2. 创建 TorchProfilerWrapper（首次）
    self.profiler = TorchProfilerWrapper(
        self.profiler_config,
        worker_name=trace_name,    # 文件名前缀
        local_rank=self.local_rank,
        activities=["CPU", "CUDA"],
    )

    # 3. 启动/停止
    self.profiler.start()  # or self.profiler.stop()
```

### Trace 文件输出（wrapper.py:192）

- 使用 `torch.profiler.tensorboard_trace_handler`
- `worker_name` 作为文件名前缀 → TP=2 产生两个文件
- 格式：`{worker_name}.{timestamp}.pt.trace.json.gz`
- 每个 rank 独立保存，不做跨 rank 聚合

### TP=2 场景下的关键差异

| 维度 | TP=1 | TP=2 |
|------|------|------|
| Trace 文件数 | 1 | 2（tp0, tp1）|
| NCCL kernel | 无 | 每层 2 次 all-reduce（attn_out, FFN_down）|
| 每个 rank 的 GEMM 尺寸 | `[T, D] × [D, N]` | `[T, D] × [D, N/2]`（权重被切半）|
| FlashAttention heads | Q=28, KV=4 | Q=14, KV=2（per rank）|

## 2. 实验设计

- 脚本：`experiments/day2_distributed_profiling.sh`
- 对比：TP=1 vs TP=2，同一 prefill 请求
- 分析脚本：`experiments/day2_analyze_distributed.py`

## 3. 实验结果

（待实验后填写）

### TP=1 组件分布

| 组件 | 时间(ms) | 占比 |
|------|----------|------|
| GEMM (Linear) | 81.16 | 86.9% |
| Attention (FlashAttn) | 6.14 | 6.6% |
| Normalization | 1.99 | 2.1% |
| Other | 1.38 | 1.5% |
| KV Cache | 0.96 | 1.0% |
| Activation | 0.88 | 0.9% |

### TP=2 组件分布（Rank 0）

| 组件 | 时间(ms) | 占比 |
|------|----------|------|
| GEMM (Linear) | 43.77 | 66.2% |
| Communication (NCCL) | 12.76 | 19.3% |
| Attention (FlashAttn) | 6.06 | 9.2% |
| Other | 1.47 | 2.2% |
| KV Cache | 0.95 | 1.4% |
| Activation | 0.61 | 0.9% |

### NCCL 通信开销（Rank 0）

| Kernel | 时间(ms) | 次数 | 平均(us) |
|--------|----------|------|----------|
| oneshotAllreduceFusionKernel | 5.460 | 855 | 6.4 |
| twoshotAllreduceKernel | 3.589 | 57 | 63.0 |
| rmsNormLamport (allreduce+RMSNorm 融合) | 3.579 | 57 | 62.8 |
| ncclDevKernel_AllGather_RING_LL | 0.133 | 16 | 8.3 |

### 对比结论

| 指标 | TP=1 | TP=2 R0 | 变化 |
|------|------|---------|------|
| Total CUDA Time | 93.4ms | 66.1ms | -29.2% |
| GEMM Time | 81.2ms | 43.8ms | -46.1% |
| FlashAttn Time | 6.1ms | 6.1ms | 不变 |
| NCCL Communication | 0 | 12.8ms | 新增 |

关键发现：
1. **GEMM 缩减比 0.54x** — 接近理论 0.5x，权重沿列/行切半
2. **NCCL 通信占 19.3%** — TP=2 的代价，每层 2 次 all-reduce
3. **FlashAttention 不变** — head 按 rank 切分，序列长度不变，单 rank 工作量相近
4. **净加速 29%** — GEMM 省下 37.4ms，被 12.8ms NCCL 通信吃掉部分
5. **Rank 间 NCCL 不对称** — Rank 0 (12.8ms) vs Rank 1 (6.9ms)，rank 0 等待同步更久
6. **allreduce+RMSNorm 融合** — `rmsNormLamport` kernel 把通信和归一化合并，减少 kernel launch

## 4. 瓶颈→决策映射框架

### 判断流程

```
Step 1: 确定场景（prefill vs decode vs mixed）
        ↓
Step 2: 运行 profiling，识别主要瓶颈
        ↓
Step 3: 按瓶颈类型查表选优化手段
        ↓
Step 4: 预估收益，决定优先级
```

### 瓶颈分类与优化映射表

| 瓶颈类型 | 判断依据 | 优化手段 | 适用场景 | 预期收益 |
|----------|----------|----------|----------|----------|
| **Compute-bound GEMM** | GEMM 占比 >80%, 计算强度 > ridge point | ① FP8 量化（减半 FLOPs）② TP 并行（切分权重）③ Speculative decode（减少 decode 步数） | Prefill 阶段, 大 batch | GEMM 时间减半（FP8）|
| **Memory-bound GEMM** | GEMM 占比高但计算强度 < ridge point | ① KV Cache 量化（减少访存）② 增大 batch（提高强度）③ Weight-only 量化（W4A16）| Decode 阶段, 小 batch | 取决于访存减少比例 |
| **Attention-bound** | FlashAttn 占比 >20% | ① 减少 KV Cache 长度（滑窗/H2O）② MQA/GQA（减少 KV heads）③ MLA（压缩 KV）④ PagedAttention 优化 | 长序列 prefill | 依赖具体序列长度 |
| **Communication-bound** | NCCL 占比 >15% | ① 减少 TP 度（如 TP=4→TP=2）② 通信计算 overlap ③ Sequence Parallelism ④ 使用更快互联（NVLink vs PCIe）| 多卡推理 | 消除通信 bubble |
| **KV Cache Memory** | OOM 或 KV cache miss rate 高 | ① KV Cache 量化（FP8/INT4）② Prefix caching ③ PagedAttention ④ 减少 num_kv_heads | 长序列/大并发 | 内存容量问题 |
| **CPU overhead** | CPU time >> CUDA time gap | ① CUDA Graph（减少 launch 开销）② 减少 Python 开销 ③ 优化 scheduler | 小 batch decode | 消除 CPU 瓶颈 |
| **Scheduling/Queueing** | TTFT 高但 kernel 利用率低 | ① Chunked prefill ② 调整 max_num_seqs ③ Priority scheduling | 混合请求场景 | 降低 TTFT |

### Roofline 快速判断规则

```
对于 GEMM kernel [M, K] × [K, N]:
  计算强度 I = 2MN / (2(MK + KN + MN))  ≈ M (当 M << K,N)
  ridge point (H800 FP16) ≈ 295 FLOPs/byte

  if I > 295: compute-bound → FP8 量化、TP 并行
  if I < 295: memory-bound → 增大 batch、weight 量化
```

```
Prefill (T=长序列):
  GEMM 强度 ∝ T → 大 T 时 compute-bound → FP8 + TP

Decode (T=1 per request, batch=B):
  GEMM 强度 ∝ B → 小 B 时 memory-bound → 增大 batch
                  → 大 B 时可能转 compute-bound
```

### 优化优先级排序原则

1. **先量化后并行** — FP8/INT8 量化是"免费"性能（精度损失可控）
2. **先 batch 后 kernel** — 调整 batch 策略收益大于优化单个 kernel
3. **先通信后计算** — 如果通信是瓶颈，优化计算没用
4. **先测量后优化** — 每次优化前后都要 profiling 验证
