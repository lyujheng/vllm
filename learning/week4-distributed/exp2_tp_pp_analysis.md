# 实验2笔记：TP=4,PP=1 vs TP=2,PP=2 在线推理性能对比

> 实验时间：2026-09-02
> 配置：Qwen2.5-7B, gpu_memory_utilization=0.85, random dataset (input_len=256, output_len=128, 100 prompts, request_rate=inf)
> 环境：8× H800 80GB, NVLink, vLLM 0.27.1 (V1 引擎, online serving)

---

## 1. 测试结果

| 指标 | TP=4, PP=1 (4卡) | TP=2, PP=2 (4卡) |
|------|-----------------|-----------------|
| Successful requests | 100 | 100 |
| Request throughput (req/s) | 111.18 | 67.77 |
| Output token throughput (tok/s) | 14,231 | 8,675 |
| Mean TTFT (ms) | 304.79 | 336.13 |
| Mean TPOT (ms) | 4.46 | 8.77 |
| Mean ITL (ms) | 5.34 | 8.77 |

## 2. 分析

### 2.1 PP 引入的性能损失

同样 4 张卡，纯 TP 方案的吞吐是 TP+PP 方案的 **1.64 倍**。

TP=2,PP=2 的 TPOT（8.77ms）接近 exp1 中 TP=1 的 TPOT（8.20ms），说明 PP=2 时 decode 速度基本由单个 stage 内的 TP=2 决定，另一个 stage 的 2 张卡在等待中没有贡献计算。

### 2.2 PP 慢的根本原因：Pipeline Bubble

推理时 PP 是逐 stage 串行执行的，不是训练中的 micro-batch 流水线。

自回归生成的特点决定了这一点：
- Step N：stage 0 → stage 1 → 得到 token N
- Step N+1：需要 token N 作为输入 → stage 0 才能开始
- 自回归依赖链使两个 stage 无法在同一个请求上流水并行

Continuous batching 虽然把多个请求打包成一个 batch，但整个 batch 一起走 stage 0 → stage 1 的串行路径，不按请求单独调度到不同 stage。

PP 性能瓶颈排序：
1. **Pipeline bubble（串行等待）** → 主要瓶颈，即使通信零延迟也无法消除
2. **P2P 通信延迟** → 次要，跨节点时更明显
3. **通信量** → 几乎不是问题（每次只传 hidden states，约几百 KB）

### 2.3 PP 通信量分析

PP 每次 stage 间传输的是 hidden states：
- shape: [batch_size, hidden_dim]
- Qwen2.5-7B: hidden_dim=3584, bf16=2 bytes
- batch_size=100 时：100 × 3584 × 2 ≈ 700 KB
- 每个 forward 只传 1 次

对比 TP 的 AllReduce：每层 2 次 × 28 层 = 56 次/forward，通信频繁得多。

所以 PP 适合跨节点（通信量小，对带宽要求低），TP 需要节点内 NVLink（通信频繁，对带宽要求高）。

## 3. 结论

### 模型放得下时，避免使用 PP

| 场景 | 推荐方案 |
|------|---------|
| 模型单卡能放下 | TP=1，用 DP 扩吞吐 |
| 模型单卡放不下 | TP 切到单卡能放，同节点内 NVLink |
| TP 切满单节点还放不下 | 跨节点加 PP（通信量小，容忍低带宽） |
| 需要最低延迟 | 尽量多用 TP（并行计算），避免 PP（串行等待） |

### TPOT vs ITL

- TPOT（Time Per Output Token）：decode 阶段平均每 token 耗时，衡量整体效率
- ITL（Inter-Token Latency）：相邻 token 间隔，衡量输出平稳性
- ITL 毛刺常见原因：chunked prefill 抢占、batch size 波动、CUDA Graph miss
- 生产中 P99 ITL 比 TPOT 均值更重要（用户体验）
