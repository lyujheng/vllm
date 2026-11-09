# Week 7 量化方案性能对比分析

模型：Qwen2.5-7B，GPU：H800 单卡，100 请求并发，input_len=256 token，output_len=128 token（随机数据，仅测计算性能）

## 1. 实验配置说明

| 配置 | 量化类型 | 量化方式 | 权重精度 | 激活精度 | Scale 粒度 | 说明 |
|------|---------|---------|---------|---------|-----------|------|
| BF16 基线 | 无 | — | BF16 | BF16 | — | 不做任何量化，参考基准 |
| FP8 per-tensor | W8A8 | 在线（online） | FP8 E4M3 | FP8 E4M3 | 整个 tensor 共享 1 个 scale | 启动时将 BF16 权重转为 FP8，forward 时激活也量化为 FP8，用 FP8 Tensor Core 计算 |
| FP8 per-block | W8A8 | 在线（online） | FP8 E4M3 | FP8 E4M3 | 每 128×128 block 1 个 scale | 同上，但 scale 粒度更细，理论精度更高，kernel 更复杂 |
| INT4 AWQ | W4A16 | 离线（offline） | INT4 | BF16 | per-group（group_size=128） | 预量化模型，权重 4-bit 存储，推理时在 kernel 内反量化为 FP16 再计算。AWQ 通过激活感知保护重要通道 |
| INT4 GPTQ | W4A16 | 离线（offline） | INT4 | BF16 | per-group（group_size=128） | 预量化模型，与 AWQ 类似但用 Hessian 矩阵做逐列误差补偿。两者都走 Marlin kernel |
| BF16 + FP8 KV | KV Cache 量化 | 在线 | BF16 | BF16 | per-tensor | 权重不量化，仅将 KV Cache 从 BF16 压缩到 FP8，每 GiB 可存更多 token。目标是扩大并发容量，不是加速计算 |

**在线 vs 离线量化**：
- 在线（online）：不需要预量化模型，vLLM 启动时用 `--quantization fp8_per_tensor` 参数，自动将 BF16 权重转换为量化格式
- 离线（offline）：需要提前用 AutoAWQ/AutoGPTQ 等工具将模型量化并保存，vLLM 直接加载预量化的 checkpoint

**per-tensor vs per-block scale**：
- per-tensor：整个权重矩阵（如 4096×4096）共用 1 个 scale 值。简单高效，但如果矩阵内数值分布不均匀，精度会损失
- per-block：每 128×128 个元素用 1 个 scale 值。精度更好，但需要更多 scale 存储和更复杂的 kernel 逻辑

## 2. 性能总览

| 配置 | 权重显存 | KV Cache | Output tok/s | Mean TTFT | Mean TPOT | P99 TPOT |
|------|:-------:|:-------:|:---:|:---------:|:---------:|:--------:|
| BF16 基线 | 14.63 GiB | 51.64 GiB | 5297 | 778.9ms | 7.57ms | 8.91ms |
| FP8 per-tensor | 8.59 GiB | 57.59 GiB | **7506** | **532.8ms** | **5.69ms** | **6.30ms** |
| FP8 per-block | 8.59 GiB | 57.61 GiB | 440 ⚠️ | 18099ms ⚠️ | 78.62ms | 125.23ms |
| INT4 AWQ | 5.76 GiB | 60.45 GiB | 6087 | 689.5ms | 6.74ms | 7.83ms |
| INT4 GPTQ | 5.69 GiB | 60.52 GiB | 6558 | 600.8ms | 6.60ms | 7.91ms |
| BF16 + FP8 KV | 15.64 GiB | 50.57 GiB | 5087 | 882.6ms | 7.63ms | 8.98ms |

## 3. 分析

### FP8 per-tensor 是 H800 上的最优方案

相比 BF16 基线：
- **吞吐提升 42%**（5297 → 7506 tok/s）
- **TPOT 降低 25%**（7.57 → 5.69ms）
- **权重显存减半**（14.63 → 8.59 GiB），多出 6 GiB 给 KV Cache

这是两方面收益叠加的结果：
1. FP8 权重体积小，访存量减少 → 单次 GEMM 更快
2. 腾出的显存给 KV Cache → 能同时服务更多请求 → 更好的 batching 效率

### FP8 per-block 性能异常

TPOT 高达 78.6ms（BF16 的 10 倍），吞吐只有 440 tok/s。从日志分析：

- CUDA graph PIECEWISE 捕获花了 **42 秒**（per-tensor 只要 0.7 秒）
- per-block 量化每个 128x128 block 有独立 scale，GEMM kernel 更复杂
- 可能触发了大量 Triton kernel JIT 编译或 CUDA graph 回退到 eager 执行

**结论**：fp8_per_block 在线量化对 Qwen2.5-7B 这种 dense 模型不适合，可能是 kernel 实现尚未充分优化，或者 block-level scaling 的开销在小模型上占比太大。

### INT4（AWQ vs GPTQ）

两者性能接近，GPTQ 略快：
- GPTQ 吞吐 6558 tok/s vs AWQ 6087 tok/s
- 权重显存几乎一致（5.69 vs 5.76 GiB）
- 都比 BF16 快（4x 压缩 → 更多 KV Cache + 更少访存），但比 FP8 per-tensor 慢

GPTQ 比 AWQ 快的原因可能是两者使用了不同的 Marlin kernel 变体。

### BF16 + FP8 KV Cache

性能反而略低于纯 BF16：
- 权重显存增加了 1 GiB（15.64 vs 14.63 GiB），可能是 FP8 KV Cache 需要额外存储 k_scale/v_scale
- KV Cache 容量反而减少（50.57 vs 51.64 GiB）

原因：`--kv-cache-dtype fp8` 将 KV Cache 从 BF16 压缩到 FP8，理论上容量翻倍。但在 gpu-memory-utilization=0.85 的限制下，KV Cache 总显存上限不变，FP8 KV Cache 的优势体现在每 GiB 能存更多 token，而不是获得更多 GiB。**当前 100 请求并发没有打满 KV Cache 容量**，所以 FP8 KV Cache 的优势没有体现出来，反而因为量化/反量化的额外开销略有劣化。

FP8 KV Cache 的真正价值在高并发 / 长上下文场景，KV Cache 成为瓶颈时才会显现。

## 4. 显存对比

```
权重占用（GiB）:
  BF16:           14.63  ████████████████
  FP8 per-tensor:  8.59  █████████
  INT4 AWQ:        5.76  ██████
  INT4 GPTQ:       5.69  ██████

KV Cache 可用（GiB）:
  BF16:           51.64  ████████████████████████████████████████████████████
  FP8 per-tensor: 57.59  █████████████████████████████████████████████████████████
  INT4 AWQ:       60.45  ████████████████████████████████████████████████████████████
  INT4 GPTQ:      60.52  ████████████████████████████████████████████████████████████
```

INT4 量化将权重压缩到 ~5.7 GiB，比 BF16 省出 ~9 GiB 给 KV Cache，在高并发场景下优势更大。

## 5. 补充实验：FP8 KV Cache 高并发长上下文验证

前面实验（exp6）用 100 请求 × 384 token 的轻负载测试 FP8 KV Cache，结果反而比 BF16 略慢。原因是 KV Cache 容量远未打满，量化/反量化的额外开销反而拖累了性能。

为验证 FP8 KV Cache 在 KV Cache 成为瓶颈时的效果，设计了压力测试：

| 参数 | 轻负载（exp6） | 压力测试（exp7） |
|------|:------------:|:--------------:|
| input_len | 256 | **4096** |
| output_len | 128 | **512** |
| num_prompts | 100 | **500** |
| max_num_seqs | 64 | **256** |
| gpu_memory_utilization | 0.85 | **0.90** |

### 结果对比

| 指标 | BF16 KV | FP8 KV | 变化 |
|------|:-------:|:------:|:----:|
| KV Cache 显存 | 55.48 GiB | 54.41 GiB | 相近 |
| 可存 token 数 | ~101 万 | ~198 万 | **1.96x** |
| Output tok/s | 3203 | **3780** | **+18%** |
| Request throughput | 6.26 req/s | **7.38 req/s** | **+18%** |
| 总耗时 | 79.9s | **67.7s** | **-15%** |
| Mean TPOT | 59.96ms | **54.17ms** | **-10%** |
| Mean TTFT | 31.0s | **28.2s** | **-9%** |

### 分析

同样的显存空间，FP8 KV Cache 每个 token 只占 1 byte（BF16 占 2 bytes），token 容量接近翻倍。调度器可以同时处理更多请求，batch 更大，GPU 利用率更高，吞吐提升 18%。

对比轻负载实验（FP8 KV 反而更慢）和压力测试（FP8 KV 快 18%），结论很清晰：

- **KV Cache 未满时**：FP8 KV Cache 没有收益，量化/反量化开销反而拖累性能
- **KV Cache 打满时**：容量翻倍 → 并发提升 → batch 效率提升 → 吞吐提升

FP8 KV Cache 的本质是**用微小的精度损失换取容量翻倍**，只有在 KV Cache 成为系统瓶颈时才有价值。典型适用场景：高并发服务、长上下文（32K-128K）、大模型显存紧张。

## 6. 总结与建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| H800 通用部署 | **FP8 per-tensor** | 吞吐最高，精度几乎无损 |
| 显存受限 / 高并发 | INT4 GPTQ/AWQ | 4x 压缩，最大化 KV Cache 空间 |
| 长上下文场景 | FP8 权重 + FP8 KV Cache | 双重压缩，最大化序列长度 |
| 追求精度 | BF16 | 无量化损失 |

**不推荐**：fp8_per_block（在线量化），当前版本性能异常，不适合生产使用。
