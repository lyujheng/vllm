# Day 5: DFlash + DSpark 实验结果

## 实验环境

- GPU: 单卡 H800 (80GB)
- Target 模型: Qwen3-8B (`/data/model/Qwen3-8B`)
- Draft 模型: dspark_qwen3_8b_block7 (`/data/model/dspark_qwen3_8b_block7`)
- 数据集: sonnet（真实文本），input_len=256, output_len=128
- 请求: 200 prompts, request_rate=10

## 实验 06: DSpark vs Baseline

### 结果对比

| 指标 | Baseline | DSpark (K=7) | 变化 |
|------|----------|-------------|------|
| Request throughput (req/s) | 9.57 | 9.77 | +2.1% |
| Output throughput (tok/s) | 1224.88 | 1250.91 | +2.1% |
| Mean TTFT (ms) | 28.12 | 36.92 | +31.3% ↑ |
| Median TTFT (ms) | 27.52 | 34.35 | +24.8% ↑ |
| P99 TTFT (ms) | 48.49 | 58.98 | +21.6% ↑ |
| Mean TPOT (ms) | 6.92 | 3.28 | **-52.6% ↓** |
| Median TPOT (ms) | 6.93 | 3.25 | **-53.1% ↓** |
| P99 TPOT (ms) | 7.11 | 3.99 | -43.9% ↓ |
| Mean ITL (ms) | 6.92 | 9.20 | +32.9% ↑ |

### DSpark 接受率

| 位置 | 接受率 |
|------|--------|
| Position 0 | 71.73% |
| Position 1 | 47.16% |
| Position 2 | 28.98% |
| Position 3 | 16.49% |
| Position 4 | 10.46% |
| Position 5 | 5.96% |
| Position 6 | 3.43% |
| **Overall** | **26.32%** |
| **平均接受长度** | **2.84** |

### 结论

**1. TPOT 降低 52.6% — 投机解码的核心价值**

decode 阶段每个 output token 的延迟从 6.92ms 降到 3.28ms。原因：每次 draft+verify 一轮接受约 2.84 个 token，摊薄了每个 token 的 target forward 开销。

**2. TTFT 增加 31.3% — 投机解码的固有代价**

从 28.12ms 增加到 36.92ms，多出约 9ms。通过代码追踪确认原因：prefill 完成后、首 token 返回前，DFlash 的完整 draft 流程被同步执行（context KV precompute + draft forward + sample），目的是为第二步 target 验证提前准备 draft tokens，避免冷启动。

代码路径：`gpu_model_runner.py` L4657-4658，target 采样后立即调用 `propose_draft_token_ids()`，在首 token 返回前完成。

**3. ITL 增加 32.9% — 突发输出模式**

从 6.92ms 增加到 9.20ms。投机解码的输出模式是突发式的：每次 verify 一轮输出多个 token，然后等待下一轮 draft+verify，导致 token 间隔不均匀。

**4. 整体吞吐提升有限 (+2.1%)**

原因有二：
- 接受率偏低（26.32%），大部分 draft 计算被浪费
- request_rate=10 的中等负载下（Peak concurrent=23-25），投机解码的优势主要体现在延迟而非吞吐

**5. 接受率分析**

Position 0 有 71.73% 的接受率，但后续位置快速衰减至 3.43%（Position 6）。这与 Qwen3-8B 的模型规模有关——8B 模型本身不大，draft 模型的预测能力受限。DSpark 在更大的 target 模型（如 72B）上 draft 与 target 的能力差距更大，接受率会更高。

## 对比：random 数据集 vs sonnet 数据集

首次使用 random 数据集的实验结果中 Position 0 接受率为 50.86%，切换到 sonnet 后提升至 71.73%。这验证了投机解码在真实文本上的优势——自然语言有局部可预测性，draft 模型能更好地预测。

## 适用场景总结

DSpark 投机解码适合：
- **延迟敏感场景**：TPOT 降低 50%+，单请求体验显著改善
- **低并发场景**：batch size 小时 target forward 是 memory-bound，draft 更多 token 能充分利用计算资源
- **大 target 模型**：draft 与 target 能力差距大，接受率更高

不适合：
- **高并发场景**：batch size 大时 target 已经 compute-bound，draft 验证开销反而拖累吞吐
- **TTFT 极致敏感**：首 token 延迟会增加（本实验 +9ms）

## 实验 07: 不同 num_speculative_tokens (K) 对比

### 结果对比

| 指标 | K=1 | K=3 | K=5 | K=7 |
|------|-----|-----|-----|-----|
| Output throughput (tok/s) | 1239.54 | 1248.44 | 1250.93 | 1252.09 |
| Mean TTFT (ms) | **1872.54** | 35.24 | 37.73 | 35.51 |
| Median TTFT (ms) | 237.66 | 32.64 | 36.04 | 34.04 |
| P99 TTFT (ms) | 8249.93 | 133.04 | 59.00 | 59.05 |
| Mean TPOT (ms) | 6.75 | 3.75 | 3.44 | **3.19** |
| Mean ITL (ms) | 11.01 | 8.75 | 9.09 | 8.88 |
| Acceptance rate (%) | 63.68 | 45.12 | 33.37 | 26.00 |
| 平均接受长度 | 1.64 | 2.35 | 2.67 | 2.82 |
| Peak concurrent | **106** | 24 | 24 | 24 |

### 逐位置接受率

| Position | K=1 | K=3 | K=5 | K=7 |
|----------|-----|-----|-----|-----|
| 0 | 63.68% | 67.02% | 69.32% | 71.41% |
| 1 | — | 42.11% | 45.04% | 46.59% |
| 2 | — | 26.25% | 27.40% | 28.52% |
| 3 | — | — | 15.59% | 16.36% |
| 4 | — | — | 9.52% | 10.09% |
| 5 | — | — | — | 5.74% |
| 6 | — | — | — | 3.29% |

### 结论

**1. K=1 是灾难性选择**

TTFT 暴涨至 1872ms（vs K=3 的 35ms），Peak concurrent 飙至 106。原因：K=1 时每步 draft 1 个 token，接受率 63.68% 意味着约 36% 的 draft 被浪费，而 draft+verify 的额外开销几乎抵消了接受 1 个 token 的收益。吞吐没有提升，请求堆积严重。

**2. TPOT 随 K 增大持续降低**

K=1 → K=3：6.75 → 3.75ms（-44.4%，最大降幅）
K=3 → K=5：3.75 → 3.44ms（-8.3%）
K=5 → K=7：3.44 → 3.19ms（-7.3%）

K=3 是收益拐点，从 K=1 到 K=3 的提升最为显著。K=5 和 K=7 的边际收益递减。

**3. 接受率随 K 增大下降，但平均接受长度增加**

| K | 接受率 | 平均接受长度 |
|---|--------|------------|
| 1 | 63.68% | 1.64 |
| 3 | 45.12% | 2.35 |
| 5 | 33.37% | 2.67 |
| 7 | 26.00% | 2.82 |

接受率下降是因为后面位置越来越难预测。但 K 越大，一轮 draft 能被接受的 token 绝对数量越多（2.82 > 2.35 > 1.64），每次 target verify 的"性价比"更高，所以 TPOT 持续降低。

**4. Position 0 接受率在不同 K 值下稳定（63-71%）**

说明 draft 模型对"下一个 token"的预测能力不受 K 值影响。K 只是决定了 draft 多远的位置，不影响近处位置的质量。

**5. K 值选择建议**

- **K=1**：不推荐，开销和收益几乎持平，可能导致请求堆积
- **K=3**：性价比最高的拐点，TPOT 大幅下降且 TTFT 正常
- **K=5**：TPOT 继续小幅改善，适合延迟敏感场景
- **K=7**：TPOT 最优但边际收益有限，draft 计算浪费较多（后三个位置接受率 <10%）
- 实际部署建议结合 Dynamic SD Schedule：低并发用大 K，高并发用小 K
