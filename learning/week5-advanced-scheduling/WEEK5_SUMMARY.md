# Week 5: 高级调度与优化 - 总结报告

**完成时间**: 2026-09-09  
**学习周期**: Week 5 Day 1-7

---

## 一、学习目标达成情况

- [x] Day 1-2: Chunked Prefill 原理与实现
- [x] Day 3: Prefix Caching (APC) 机制  
- [x] Day 4-5: Speculative Decoding 完整流程
- [x] Day 6: CUDA Graph 深入（两种实现方式）
- [x] Day 7: 综合实验与性能对比

---

## 二、核心知识点总结

### 1. Chunked Prefill ✅

**原理**：
- 问题：长 prompt prefill 阻塞 decode，影响在线延迟
- 方案：将 prefill 拆成多个 chunk，与 decode 交替执行
- 关键参数：`max_num_batched_tokens` 控制 chunk 大小

**代码位置**：
- `vllm/v1/core/sched/scheduler.py` - `_schedule_prefills()`

**实验结果**（exp1: Chunked Prefill，模型 Qwen2.5-7B，input_len=2048，output_len=128，30 prompts）：
```
配置                    | Req Throughput | Mean TTFT  | P99 TTFT   | Mean ITL | P99 ITL
无 Chunked Prefill      | 4.38 req/s    | 88.01 ms   | 181.39 ms  | 7.19 ms  | 35.90 ms
Chunked 8192            | 4.38 req/s    | 86.33 ms   | 171.59 ms  | 7.28 ms  | 44.05 ms
Chunked 2048            | 4.37 req/s    | 94.32 ms   | 184.64 ms  | 7.31 ms  | 34.62 ms
Chunked 1024            | 4.38 req/s    | 94.88 ms   | 183.54 ms  | 7.08 ms  | 22.32 ms
```

**分析**：
- Chunk 越小，P99 ITL 越低（35.90 → 22.32 ms，↓37.8%），decode 被阻塞的时间更短
- 但 TTFT 略有增加（88.01 → 94.88 ms），因为 prefill 需要更多轮才能完成
- 吞吐量基本不变（~4.38 req/s），chunk 主要影响延迟分布而非总吞吐
- **权衡**：小 chunk 利于在线服务（低尾延迟），大 chunk 利于快速完成 prefill

---

### 2. Prefix Caching (APC) ✅

**原理**：
- 相同 prefix 的请求共享 KV cache blocks（hash 匹配）
- 命中时直接复用 blocks，跳过 prefill
- LRU 淘汰策略

**代码位置**：
- `vllm/v1/core/block_pool.py` - `_get_cached_block()`
- `request.update_block_hashes()`

**适用场景**：
- System prompt 共享
- Few-shot examples
- RAG context

**实验结果**（exp2: Prefix Caching，模型 Qwen2.5-7B，random dataset，random_prefix_len=256，30 prompts）：
```
配置                    | Req Throughput | Mean TTFT  | P99 TTFT   | Mean ITL | P99 ITL
Prefix Caching OFF      | 1.84 req/s    | 101.66 ms  | 174.49 ms  | 6.51 ms  | 36.11 ms
Prefix Caching ON       | 1.85 req/s    | 77.74 ms   | 115.83 ms  | 6.29 ms  | 24.80 ms
```

**分析**：
- Mean TTFT 降低 23.5%（101.66 → 77.74 ms），prefix 部分无需重新计算
- P99 TTFT 降低 33.6%（174.49 → 115.83 ms），尾延迟改善更明显
- P99 ITL 降低 31.3%（36.11 → 24.80 ms）
- 吞吐量基本持平，prefix caching 主要减少重复计算的延迟

---

### 3. Speculative Decoding ✅

**原理**：
- Draft model 生成 K 个候选 tokens
- Target model 并行验证
- Rejection sampling 决定接受/拒绝

**关键组件**：
- `sample_recovered_tokens()`: 预采样 backup tokens
- `rejection_random_sample_kernel`: GPU kernel 执行 accept/reject
- KV Cache 共享：draft 和 target 共用物理 tensor，通过 `num_lookahead_tokens` 预留空间

**代码位置**：
- `vllm/v1/sample/rejection_sampler.py`
- `vllm/v1/spec_decode/llm_base_proposer.py`

**执行流程**：
```
Scheduler → Target Model Forward → Sample → Draft Model Propose → Rejection Sampling → 返回 1~K+1 tokens
```

**实验结果**：
- 未执行 Speculative Decoding 实验（需要额外的 draft model）
- 理论加速比取决于 acceptance rate 和 K 值

---

### 4. CUDA Graph ✅

**两种实现方式**：

| 特性 | CUDAGraphWrapper | BreakableCUDAGraphWrapper |
|-----|------------------|---------------------------|
| 切分时机 | 编译时（torch.compile FX splitting） | 运行时（capture 中断） |
| 依赖 | torch.compile + Inductor | 不依赖 torch.compile |
| 开启方式 | 默认（`use_inductor_graph_partition=True`） | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| Split points | `_attention_ops` 列表 | `@eager_break_during_capture` 装饰器 |

**切分原因**：
- Attention 算子访问动态地址的 KV Cache，无法录制
- 在 attention 处切分：录制前后的 matmul 等固定 shape 操作

**多 stream 协调**：
- vLLM 使用多个 stream：`compute_stream`, `copy_stream`, `comm_stream`, `broadcast_stream`
- CUDA Graph 只录制 `main_stream` 上的计算
- 其他 stream 通过 `sync_prev_onload()` / `join_after_forward()` 同步
- DMA Engine / Copy Engine 不占用 SM，可与计算真正并行

**实验结果**（exp3: CUDA Graph，模型 Qwen2.5-7B，单请求 TPOT 测试，10 轮）：
```
配置                    | 平均 TPOT
enforce_eager (无 Graph) | 203.86 ms/token
CUDA Graph (默认)        | 130.01 ms/token
```

**分析**：
- CUDA Graph 使 TPOT 降低 36.2%（203.86 → 130.01 ms）
- 减少了 kernel launch overhead 和 Python→GPU 的调度开销
- 代价：warmup 时间更长（需要 capture 多组 batch size 的 graph）

---

## 三、实验结果汇总

### 实验环境
- GPU: NVIDIA A800 80GB
- 模型: Qwen2.5-7B（/data/model/Qwen2.5-7B）
- vLLM 版本: v1 架构（0.x dev）

### 组合优化性能对比（exp5: 4 个请求，串行）

| 配置 | 平均吞吐 (tok/s) | 相对加速 | 初始化时间 (s) |
|-----|-----------------|---------|--------------|
| Baseline (enforce_eager) | 115.89 | 1.00x | 17.60 |
| + CUDA Graph | 166.17 | 1.43x | 32.27 |
| + CUDA Graph + Prefix Caching | 166.46 | 1.44x | 17.49 |
| + 所有优化 (Graph+Prefix+Chunked) | 167.74 | 1.45x | 32.36 |

### 关键发现

1. **CUDA Graph 收益最大**：单项优化即可提升 43% 吞吐，36% TPOT 降低。主要消除 kernel launch 和 Python 调度开销。
2. **Prefix Caching 优化延迟**：TTFT 降低 23.5%，P99 TTFT 降低 33.6%。对吞吐影响小，但对用户体验（首 token 等待时间）影响大。
3. **Chunked Prefill 优化尾延迟**：P99 ITL 降低 37.8%（chunk=1024），代价是 TTFT 略增。对在线服务的 SLA 保障最重要。
4. **组合优化接近线性叠加**：三个优化维度基本正交，可以同时开启。
5. **CUDA Graph 的初始化代价**：warmup 时间从 17.6s 增到 32.3s，适合长期运行的服务。

---

## 四、代码路径速查

```
vllm/v1/core/sched/scheduler.py          # Scheduler 核心逻辑
vllm/v1/core/block_pool.py               # KV Cache 管理、Prefix Caching
vllm/v1/spec_decode/                     # Speculative Decoding
vllm/v1/sample/rejection_sampler.py      # Rejection Sampling
vllm/compilation/cuda_graph.py           # CUDAGraphWrapper
vllm/compilation/breakable_cudagraph.py  # BreakableCUDAGraphWrapper
vllm/compilation/backends.py             # VllmBackend (torch.compile)
vllm/config/compilation.py               # 编译配置、splitting_ops
```

---

## 五、实践经验

### 优化选择建议
- **在线服务（低延迟优先）**：Chunked Prefill + CUDA Graph + Prefix Caching
- **离线批处理（高吞吐优先）**：Full CUDA Graph + 大 batch size
- **对话场景**：Prefix Caching（system prompt 共享）
- **代码生成**：Speculative Decoding（如果有合适的 draft model）

### 遇到的问题
1. **vllm serve 进程清理不彻底**：使用 `| tee` 管道时，`$!` 获取的是 tee 的 PID 而非 vllm serve 的 PID，导致 `kill` 无法正确终止 server 和 EngineCore 子进程
2. **`--disable-log-requests` 参数不兼容**：当前 vLLM 版本不支持该参数，导致 server 启动失败

### 解决方案
1. 使用 `> log 2>&1 &` 重定向替代 `| tee`，确保 `$!` 获取正确的进程 PID；清理逻辑改为 `kill` + `wait` + `sleep 10`
2. 移除不兼容参数，启动前清理旧日志避免误匹配

---

## 六、下一步计划

- [ ] Week 6: MoE 模型推理优化
  - Day 1-2: MoE 架构原理与推理挑战
  - Day 3-4: FusedMoE Kernel 深入
  - Day 5-6: Expert Parallelism
  - Day 7: MoE 模型部署实践

---

**备注**：
- 本周学习重点：理解 vLLM 核心优化技术的原理和实现
- 实验脚本位置：`learning/week5-advanced-scheduling/exp*.sh`
- 实验结果位置：`learning/week5-advanced-scheduling/results/`
