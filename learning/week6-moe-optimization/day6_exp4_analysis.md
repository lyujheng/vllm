# Day 6: EP vs TP 实验分析报告

**日期**：2026-09-16
**实验脚本**：`exp4_ep_vs_tp.sh`
**模型**：Qwen2-57B-A14B-Instruct（64 experts, top-8）
**硬件**：单机 8 卡 H800，使用 GPU 1,2,3,4
**Benchmark 参数**：50 prompts, input_len=512, output_len=128, request_rate=inf

---

## 一、指标含义速查

### 延迟类指标（单个请求体验）

| 指标 | 含义 | 反映什么 | 受什么影响 |
|------|------|---------|-----------|
| **Mean TTFT** | 用户发出请求到收到第一个 token 的时间 | Prefill 阶段性能 | prompt 长度、模型大小、并发数、是否 chunked prefill |
| **P99 TTFT** | 99% 的请求 TTFT 都低于这个值 | 最差情况的首 token 延迟 | 排队等待、prefill 与其他请求的竞争 |
| **Mean TPOT** | decode 阶段平均每个输出 token 的耗时 | Decode 阶段性能 | 并发 batch 大小、模型大小、TP 通信开销 |
| **P99 ITL** | 99% 的 token 间延迟都低于这个值 | Decode 尾部延迟稳定性 | prefill-decode 干扰、batch 大小波动、通信毛刺 |

### 吞吐类指标（系统整体能力）

| 指标 | 含义 | 反映什么 | 受什么影响 |
|------|------|---------|-----------|
| **Output throughput** | 每秒生成的输出 token 数（只算生成） | 系统生成速度，用户感知最直接 | GPU 算力、并发数、decode 效率 |
| **Total throughput** | 每秒处理的总 token 数（输入+输出） | 系统总处理能力，含 prefill 消耗的算力 | prefill 效率 + decode 效率的综合 |

**关键关系**：延迟和吞吐是矛盾的。并发越高，吞吐越大，但每个请求的延迟也越高。

---

## 二、实验结果汇总

| 指标 | Test 1: TP=4 | Test 2: TP=4+EP | Test 3: TP=2,DP=2+EP |
|------|-------------|----------------|---------------------|
| Output throughput (tok/s) | **2081.68** | 2040.84 | 1774.30 |
| Total throughput (tok/s) | **10408.38** | 10204.20 | 8871.52 |
| Mean TTFT (ms) | 1008.95 | **894.81** | 1353.66 |
| P99 TTFT (ms) | 1472.57 | **1321.40** | 1578.71 |
| Mean TPOT (ms) | **16.08** | 17.36 | 17.55 |
| P99 ITL (ms) | 156.50 | 180.45 | **77.48** |

---

## 三、对比分析

### 3.1 TP=4 vs TP=4+EP（Test 1 vs Test 2）

**现象**：
- 吞吐几乎相同（2081 vs 2040 tok/s），差距约 2%
- TTFT 降低（1008 → 894ms），EP 的 prefill 更快

**原因**：
- 纯 TP=4 模式：每个 expert 被 tensor 切成 4 片，MoE 计算后需要 AllReduce 合并
- TP=4 + EP 模式（dp_size=1）：走 NoDPEP 路径，每卡持有完整的 16 个 expert，MoE 层无需跨卡拼结果
- EP 消除了 MoE 层的 AllReduce，但 prefill 阶段改进更明显（计算密集），decode 阶段差异不大

**结论**：在 DP=1 的纯 EP 场景，EP 对 TTFT 有改善，对 decode 吞吐影响不大。

### 3.2 TP=4+EP vs TP=2,DP=2+EP（Test 2 vs Test 3）

**现象**：
- 吞吐下降（2040 → 1774 tok/s），下降约 13%
- TTFT 变差（894 → 1353ms）

**原因**：
- DP=2 引入了跨 DP rank 的 AllGatherv + ReduceScatterv 通信开销
- 50 个请求的小规模测试无法充分发挥 DP 的并发优势
- DP=2 意味着两个 engine 分别只有 TP=2，非 MoE 层的 TP 通信不变，但每卡承担更多参数

**结论**：小请求量下 DP 的通信代价 > 并发收益。DP 的优势需要高 QPS 场景才能体现。

### 3.3 P99 ITL 的反差（Test 3 远优于 Test 1/2）

**现象**：
- Test 3 的 P99 ITL（77ms）远低于 Test 1（156ms）和 Test 2（180ms）

**原因**：
- DP=2 将 50 个请求分到两个 engine，每个 engine 大约处理 25 个请求
- decode batch 更小 → 每步计算更快更稳定
- prefill-decode 混合调度时，小 batch 下 prefill 对 decode 的干扰更小
- 尾部延迟的毛刺减少

**结论**：DP 的核心价值之一是提升延迟稳定性，即使总吞吐不高，每个 engine 的延迟波动更小。

---

## 四、核心结论

1. **EP vs TP**：EP 对 MoE 模型的 prefill 有明确改善（避免 MoE 层 AllReduce），对 decode 影响不大
2. **DP 的双面性**：DP 降低单 engine 负载带来延迟稳定性提升，但引入跨 DP 通信开销，需要高 QPS 才能发挥吞吐优势
3. **延迟 vs 吞吐**：没有银弹，需要根据场景选择——低延迟选 TP+EP，高吞吐选 DP+EP

---

## 五、与 Day 5 理论的对应

| Day 5 理论 | 实验验证 |
|-----------|---------|
| EP 模式下 MoE 层无需 AllReduce | Test 2 TTFT 优于 Test 1 ✓ |
| DP+EP 引入 AllGatherv + ReduceScatterv 通信 | Test 3 吞吐低于 Test 2 ✓ |
| DP 将请求分散到多个 engine | Test 3 P99 ITL 最低 ✓ |
| ep_size = tp_size × dp_size | Test 3: EP=4 (2×2) ✓ |
