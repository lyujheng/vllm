# Day 7: MoE 部署调优实验分析报告

**日期**：2026-09-16
**实验脚本**：`exp5_moe_tuning.sh`
**模型**：Qwen2-57B-A14B-Instruct（64 experts, top-8），TP=4
**硬件**：单机 8 卡 H800，使用 GPU 1,2,3,4
**Benchmark 参数**：100 prompts, input_len=512, output_len=128, request_rate=inf

---

## 一、实验配置

| 测试 | gpu_memory_utilization | max_num_seqs | 额外参数 |
|------|----------------------|-------------|---------|
| Test 1: Baseline | 0.85 | 32 | — |
| Test 2: High Memory | 0.90 | 64 | — |
| Test 3: Chunked Prefill | 0.85 | 32 | --enable-chunked-prefill |

**注意**：这三个配置是 vLLM 通用部署参数，非 MoE 特有。MoE 相关的实验在 exp4（EP vs TP）。

---

## 二、实验结果汇总

| 指标 | Test 1: Baseline | Test 2: High Memory | Test 3: Chunked Prefill |
|------|-----------------|--------------------|-----------------------|
| Output throughput (tok/s) | 1463.73 | **1796.85** | 1573.40 |
| Total throughput (tok/s) | 7318.65 | **8984.24** | 7867.01 |
| Mean TTFT (ms) | 4029.20 | **2557.85** | 2706.89 |
| P99 TTFT (ms) | 7749.07 | **4755.20** | 7130.75 |
| Mean TPOT (ms) | **13.38** | 21.79 | 15.18 |
| P99 ITL (ms) | **45.69** | 197.38 | 95.28 |

---

## 三、分析

### 3.1 High Memory（Test 2）：吞吐最高，延迟稳定性最差

- 吞吐提升 23%（1463 → 1796 tok/s），TTFT 降低 36%（4029 → 2557ms）
- 但 Mean TPOT 从 13.38 涨到 21.79ms，P99 ITL 从 45 涨到 197ms
- 原因：mem=0.90 + seqs=64 → 更多 KV cache + 更大并发 batch → 吞吐高，但每个 decode step 处理的 token 更多，单 token 延迟上升

### 3.2 Chunked Prefill（Test 3）：TTFT 和 ITL 的折中

- TTFT 比 Baseline 好很多（4029 → 2706ms），接近 High Memory
- P99 ITL（95ms）比 High Memory（197ms）好很多，但比 Baseline（45ms）差
- 原因：chunked prefill 把长 prefill 切成小块和 decode 交替执行，降低 prefill 排队时间，同时控制对 decode 的干扰

### 3.3 Baseline（Test 1）：延迟最稳定，吞吐最低

- TPOT 和 P99 ITL 都最低（13.38ms / 45ms）
- max_num_seqs=32 限制了并发，batch 小、干扰少、decode 稳定
- 代价是吞吐低、TTFT 高（请求排队等久）

---

## 四、核心结论

没有最优配置，只有最适合场景的配置：
- **追求吞吐**（批处理、离线任务）→ High Memory 配置（大显存、高并发）
- **追求低延迟**（实时对话）→ Baseline 配置（小 batch、稳定 decode）
- **折中方案**（在线服务、兼顾首 token 和流式体验）→ Chunked Prefill
