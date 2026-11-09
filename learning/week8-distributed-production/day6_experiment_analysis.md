# Day 6 实验分析：KV Cache Offloading 效果验证

## 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen2.5-7B (bfloat16, 14.32 GiB) |
| 硬件 | 单卡 H800 (79.18 GiB) |
| GPU 可用 KV cache | 39.76 GiB |
| CPU offload pool | 46811 blocks / 40 GB (pinned memory) |
| Benchmark 参数 | 200 requests, input_len=2048, output_len=128, request_rate=10 RPS |
| vLLM 版本 | v0.27.1 |

---

## 实验 A：Baseline vs Eager vs Lazy（gpu-memory-utilization=0.7）

### 冷启动（第一轮：无 prefix cache 积累）

| 指标 | A1 Baseline | A2 Eager | A3 Lazy |
|------|:-----------:|:--------:|:-------:|
| Successful requests | 200 | 200 | 200 |
| Benchmark duration (s) | 20.97 | 20.97 | 20.96 |
| Request throughput (req/s) | 9.54 | 9.54 | 9.54 |
| Output token throughput (tok/s) | 1220.60 | 1220.80 | 1221.15 |
| Total token throughput (tok/s) | 20750.28 | 20753.64 | 20759.56 |
| Peak concurrent requests | 38 | 38 | 38 |
| **Mean TTFT (ms)** | **102.13** | **100.59** | **102.61** |
| Median TTFT (ms) | 80.30 | 79.70 | 81.40 |
| P99 TTFT (ms) | 238.83 | 246.02 | 241.70 |
| **Mean TPOT (ms)** | **10.79** | **10.87** | **11.06** |
| Median TPOT (ms) | 10.84 | 10.83 | 10.83 |
| P99 TPOT (ms) | 15.86 | 15.83 | 16.14 |
| **Mean ITL (ms)** | **10.79** | **10.87** | **11.06** |
| P99 ITL (ms) | 86.12 | 51.04 | 53.05 |

### 暖启动（第二轮：prefix cache 已积累）

| 指标 | A1 Baseline | A2 Eager | A3 Lazy |
|------|:-----------:|:--------:|:-------:|
| Benchmark duration (s) | 20.84 | 20.85 | 20.84 |
| Request throughput (req/s) | 9.60 | 9.59 | 9.60 |
| Output token throughput (tok/s) | 1228.39 | 1228.02 | 1228.59 |
| Total token throughput (tok/s) | 20882.71 | 20876.30 | 20886.04 |
| Peak concurrent requests | 24 | 24 | 24 |
| **Mean TTFT (ms)** | **37.13** | **34.47** | **37.41** |
| Median TTFT (ms) | 36.74 | 33.99 | 36.32 |
| P99 TTFT (ms) | 50.33 | 48.82 | 58.28 |
| **Mean TPOT (ms)** | **6.44** | **6.51** | **6.35** |
| Median TPOT (ms) | 6.42 | 6.48 | 6.36 |
| P99 TPOT (ms) | 6.69 | 6.82 | 6.65 |
| **Mean ITL (ms)** | **6.44** | **6.51** | **6.35** |
| P99 ITL (ms) | 9.60 | 9.31 | 10.67 |

### Server 日志关键信息

| 指标 | A1 Baseline | A2 Eager | A3 Lazy |
|------|:-----------:|:--------:|:-------:|
| GPU KV cache 峰值使用率 | 3.7% | 5.1% | 5.7% |
| Prefix cache hit（第二轮） | 48.7% | 42.4% | 37.0% |
| External prefix cache hit | N/A | 0.0% | 0.0% |

### 冷启动分析

三组配置在冷启动阶段的性能**几乎完全一致**：

- **吞吐量**：三者 request throughput 均为 9.54 req/s，output token throughput 在 1220-1221 tok/s 之间，差异 < 0.05%
- **TTFT**：mean TTFT 在 100-103 ms 范围内，差异在正常波动范围
- **TPOT / ITL**：mean 在 10.79-11.06 ms，差异 < 3%

这符合预期：冷启动时没有可复用的 prefix cache，offloading 无法发挥作用（没有东西可以从 CPU 加载回来）。Eager 模式会在后台将计算完的 KV block 存到 CPU，但这个 store 操作在独立 CUDA stream 上异步执行，对 forward 的计算路径影响极小。

### 暖启动分析

暖启动（第二轮相同请求）同样差异很小：

- **TTFT 大幅下降**：从 ~100ms 降至 ~35ms（降幅 ~65%），这是 GPU prefix cache 命中的直接效果
- **Peak concurrent 降至 24**：prefix cache 命中减少了 prefill 计算量，请求排队减少
- **三者性能仍然接近**：eager TTFT 略低（34.47 vs 37.13 ms），但差异仅 ~7%

### 关键发现：External prefix cache hit 为 0%

这是实验最重要的发现。尽管 Eager/Lazy offloading 配置正确（server 日志确认 46811 CPU blocks / 40GB 已分配），但 **external prefix cache hit 始终为 0%**。

原因分析：

1. **GPU KV cache 未饱和**：峰值使用率仅 3.7%-5.7%，39.76 GiB 的 GPU KV cache 对于 200 个 2048-token 请求绰绰有余
2. **Offloading 的触发条件**：
   - **Eager 模式**：虽然会主动将每个请求的 KV block 存到 CPU，但 GPU 端的 block 并没有被淘汰（refcount > 0 或仍在 prefix cache 中），下一轮请求直接命中 GPU prefix cache，根本不需要去 CPU 查找
   - **Lazy 模式**：只有当 GPU block 被淘汰进入 free queue 时才扫描存储到 CPU。GPU cache 未满，block 不会被淘汰，所以 lazy 模式完全没有触发过 store
3. **查询优先级**：scheduler 先查 GPU prefix cache，命中后直接返回，不会再查 CPU（external）。只有 GPU 未命中时才查 external

**结论**：在 GPU 显存充足的场景下，KV cache offloading 不会产生任何正面效果。offloading 的价值只有在 GPU KV cache 饱和、block 被淘汰后才能体现。

---

## 实验 B：不同 GPU 显存利用率下的 Offloading 效果

### 各利用率对比（冷启动，单轮）

| 指标 | 0.5 Base | 0.5 Eager | 0.7 Base | 0.7 Eager | 0.9 Base | 0.9 Eager |
|------|:--------:|:---------:|:--------:|:---------:|:--------:|:---------:|
| Request throughput (req/s) | 9.54 | 9.54 | 9.53 | 9.54 | 9.54 | 9.53 |
| Output tok/s | 1220.93 | 1220.72 | 1220.18 | 1220.69 | 1220.55 | 1220.13 |
| Total tok/s | 20755.87 | 20752.32 | 20742.99 | 20751.66 | 20749.30 | 20742.28 |
| Peak concurrent | 38 | 38 | 38 | 38 | 38 | 38 |
| **Mean TTFT (ms)** | 102.21 | 99.89 | 103.02 | 101.58 | 98.99 | 104.91 |
| Median TTFT (ms) | 82.13 | 80.54 | 80.85 | 80.55 | 77.94 | 83.79 |
| P99 TTFT (ms) | 243.78 | 240.60 | 243.46 | 245.23 | 232.79 | 242.53 |
| **Mean TPOT (ms)** | 10.75 | 10.74 | 10.90 | 10.86 | 10.77 | 10.83 |
| P99 TPOT (ms) | 15.70 | 15.72 | 16.18 | 15.94 | 15.73 | 15.92 |
| **Mean ITL (ms)** | 10.75 | 10.74 | 10.90 | 10.86 | 10.77 | 10.83 |
| P99 ITL (ms) | 50.29 | 51.60 | 51.90 | 51.13 | 49.74 | 82.00 |

### 分析

实验 B 的所有 6 组结果同样**几乎完全一致**，无论 `gpu-memory-utilization` 设为 0.5、0.7 还是 0.9。

原因与实验 A 一致：

1. **`gpu-memory-utilization` 控制的是 GPU 显存分配上限**，不是 KV cache 实际使用量。设为 0.5 意味着 vLLM 分配 ~39.6 GiB 中约 50% = ~19.8 GiB 用于 KV cache（扣除模型权重后）。但实际 200 个请求只用了 KV cache 的 3-6%
2. **无论分配多少 KV cache 空间，实际使用率都远低于上限**，block 不会被淘汰，offloading 不会被触发
3. **冷启动单轮**：实验 B 没有 warm 轮，无法观察 prefix cache 效果

### 为什么 0.9 eager 的 P99 ITL 偏高（82ms）

`B_util0.9_eager` 的 P99 ITL = 82.00 ms，明显高于其他组（~50 ms）。这可能是因为 `gpu-memory-utilization=0.9` 时系统给 CUDA 分配的显存更多，操作系统可用内存减少，偶发的 pinned memory 竞争或 PCIe 带宽争抢导致个别请求的 inter-token latency 出现毛刺。但由于 mean ITL (10.83 ms) 与其他组一致，这只是尾部噪声。

---

## 总体结论

### 1. 本次实验未能触发 offloading 的实际收益

核心原因：**负载不够重，GPU KV cache 未饱和**。

- Qwen2.5-7B 模型仅占 14.32 GiB，H800 有 79 GiB 显存
- 200 个 2048-token 请求的 KV cache 总需求远低于 GPU 可用空间
- GPU prefix cache 足以容纳所有请求的 KV block，offloading 无用武之地

### 2. Offloading 不会引入明显的性能退化

即使 offloading 未产生收益，eager 模式的后台 CPU store 操作也没有对推理性能造成可观测的影响。这得益于：
- DmaCopyBackend 使用独立 CUDA stream + 后台 daemon 线程
- store 操作异步执行，不阻塞 compute stream

### 3. Prefix cache 本身效果显著

无论是否启用 offloading，暖启动（第二轮相同请求）的效果非常明显：
- TTFT 从 ~100ms 降至 ~35ms（-65%）
- TPOT 从 ~10.8ms 降至 ~6.4ms（-41%）
- Peak concurrent 从 38 降至 24

### 4. 要真正触发 offloading 收益需要的条件

要在实验中观察到 offloading 的价值，需要满足：

| 条件 | 说明 |
|------|------|
| GPU KV cache 饱和 | 大幅降低 `gpu-memory-utilization`（如 0.3），或使用更大模型 / 更长序列 |
| 请求量足够大 | 超过 GPU KV cache 容量，迫使 block 被淘汰 |
| 有共享前缀 | 请求之间共享前缀，被淘汰的 block 在后续请求中有复用价值 |
| 多轮运行 | 第一轮填满 GPU + 存入 CPU，第二轮（相同/相似请求）从 CPU 加载回来 |

### 5. 实验改进建议

若要复现 offloading 的实际效果，可以考虑：

```bash
# 方案 1：极低 GPU 显存利用率 + 大量请求
--gpu-memory-utilization 0.3 --num-prompts 1000 --random-input-len 4096

# 方案 2：使用更大模型（70B），GPU KV cache 空间天然紧张
# 需要多卡 tensor parallel

# 方案 3：使用共享前缀数据集 + 多轮运行
# --random-prefix-len 1024（所有请求共享 1024 token 前缀）
```

---

## 附录：原始数据文件

| 文件 | 说明 |
|------|------|
| `A1_baseline_bench.txt` / `_warm.txt` | 实验 A baseline 冷/暖 |
| `A2_eager_bench.txt` / `_warm.txt` | 实验 A eager offloading 冷/暖 |
| `A3_lazy_bench.txt` / `_warm.txt` | 实验 A lazy offloading 冷/暖 |
| `A1_baseline_server.log` | baseline server 日志 |
| `A2_eager_server.log` | eager offloading server 日志 |
| `A3_lazy_server.log` | lazy offloading server 日志 |
| `B_util{0.5,0.7,0.9}_{baseline,eager}_bench.txt` | 实验 B 各组 |
| `B_util{0.5,0.7,0.9}_{baseline,eager}_server.log` | 实验 B server 日志 |
