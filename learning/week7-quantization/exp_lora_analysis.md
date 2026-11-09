# Week 7 LoRA 并发服务实验结果分析

## 1. 基线对比：无 LoRA vs 启用 LoRA

| 指标 | 实验1: 无LoRA | 实验2: 单adapter | 开销 |
|------|:---:|:---:|:---:|
| Mean TTFT (ms) | 433.2 | 479.3 | +10.6% |
| Mean TPOT (ms) | 6.7 | 8.2 | +23.1% |
| P99 TTFT (ms) | 441.6 | 493.5 | +11.8% |
| P99 TPOT (ms) | 7.0 | 8.7 | +23.5% |

**结论**：启用 LoRA 后 TTFT 增加约 11%，TPOT 增加约 23%。TPOT 开销更大，因为每个 decode step 都要额外做 Punica 的 shrink + expand 两次 GEMM。TTFT 开销相对小，因为 prefill 本身计算量大，LoRA 的增量占比低。

## 2. 双 adapter 注册 vs 单 adapter

均为 `vllm bench serve` 标准测试，每次只向一个 adapter 发请求。同一服务实例下按顺序执行 3a → 3b → 3c → 3d。

| 顺序 | 实验 | adapter | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
|:---:|------|---------|:---:|:---:|:---:|
| 第1次 | 3a | reasoning | 589.4 | 613.3 | 7.9 |
| 第2次 | 3b | rp | 459.8 | 511.7 | 8.3 |
| 第3次 | 3c | rp | 220.9 | 238.0 | 7.9 |
| 第4次 | 3d | reasoning | 242.9 | 262.1 | 7.8 |

**发现：服务预热对 TTFT 影响显著。** 第 1 次跑（3a）TTFT 最高 589ms，逐次降低，到第 3、4 次稳定在 220~240ms。预热过程包括 CUDA graph 捕获、显存池初始化等。预热完成后两个 adapter 的 TTFT 几乎一致（220 vs 242ms），说明 adapter 本身无性能差异。

**TPOT 不受预热影响**，始终在 7.8~8.3ms，因为 decode kernel 在首次调用后即稳定。

**结论**：注册多个 adapter（max-loras=2）但只用其中一个时，预热后性能与单 adapter 一致。GPU 预分配了 2 个槽位的显存，但不影响计算性能，因为 Punica kernel 只计算实际激活的 adapter。做性能对比时需注意预热效应，应丢弃首轮数据或增加 warmup 轮次。

## 3. 混合并发：max-loras=2 vs max-loras=1（核心实验）

### 实验4: max-loras=2（两个 adapter 可同时在 GPU 上）

| adapter | Mean TTFT | P50 TTFT | P99 TTFT | Mean TPOT |
|---------|:---------:|:--------:|:--------:|:---------:|
| adapter-reasoning | 414.9ms | 168.2ms | 999.0ms | 8.7ms |
| adapter-rp | 415.7ms | 168.3ms | 995.7ms | 8.4ms |

总体吞吐: 4794.6 tok/s

### 实验5: max-loras=1（GPU 只能放 1 个 adapter，需要换入换出）

| adapter | Mean TTFT | P50 TTFT | P99 TTFT | Mean TPOT |
|---------|:---------:|:--------:|:--------:|:---------:|
| adapter-reasoning | 240.2ms | 241.6ms | 250.5ms | 7.8ms |
| adapter-rp | 1279.5ms | 1280.5ms | 1283.4ms | 7.9ms |

总体吞吐: 4949.5 tok/s

### 关键发现

**max-loras=1 时，adapter-rp 的 TTFT 高达 1279ms，是 adapter-reasoning 的 5.3 倍。**

原因：调度器限制每个 batch 只能有 1 种 adapter。100 个请求交替发送两种 adapter，调度器优先处理 adapter-reasoning，adapter-rp 的请求被排队等待，直到 reasoning 的请求处理完或 batch 切换时才能调度。

max-loras=2 时，两种 adapter 可以在同一 batch 共存，TTFT 分布均匀（P50 都在 168ms），无明显偏差。

**TPOT 对比**：max-loras=1 的 TPOT 反而略低（7.8 vs 8.7ms），因为每个 batch 只有一种 adapter，Punica kernel 不需要做多 adapter 的 token 排序和间接寻址。

## 4. 总结

1. **LoRA 开销**：TTFT +11%，TPOT +23%（相对无 LoRA 基线）
2. **多 adapter 注册不影响单 adapter 性能**（只占显存不占计算）
3. **max-loras 是多租户场景的关键参数**：
   - 设太小 → 部分 adapter 请求排队，TTFT 严重劣化
   - 设太大 → GPU 显存占用增加（每个槽位预分配 LoRA 权重空间）
4. **生产建议**：max-loras 设为预期同时活跃的 adapter 数量，max-cpu-loras 设更大值用于 LRU 缓存
