# Day 2: MoE vs Dense 性能分析

## 实验结果

### Dense Model (Qwen2.5-7B)
| 指标 | 值 |
|-----|-----|
| Throughput (tok/s) | 4402.86 |
| Request throughput (req/s) | 34.40 |
| Mean TTFT (ms) | 1097.94 |
| P99 TTFT (ms) | 2066.16 |
| Mean TPOT (ms) | 8.84 |
| P99 TPOT (ms) | 11.20 |
| GPU Config | Single GPU 5 |
| Test Duration (s) | 2.91 |

### MoE Model (Qwen2-57B-A14B-Instruct)
| 指标 | 值 |
|-----|-----|
| Throughput (tok/s) | 1536.22 |
| Request throughput (req/s) | 12.00 |
| Mean TTFT (ms) | 3090.71 |
| P99 TTFT (ms) | 5804.76 |
| Mean TPOT (ms) | 26.62 |
| P99 TPOT (ms) | 32.93 |
| GPU Config | TP=2, GPU 5-6 |
| Test Duration (s) | 8.33 |
| Num Experts | 64 |
| Top-K | 8 |
| Activated Params | ~14B |

## 对比分析

### 1. 参数量 vs 计算量
- **MoE 总参数**：~57B（所有 expert）
- **MoE 激活参数**：~14B（top-8 / 64）
- **Dense 参数**：7B
- **结论**：MoE 参数是 Dense 的 8 倍，但激活参数仅 2 倍

### 2. 内存占用
- **实验配置差异**：Dense 单卡 vs MoE TP=2（两卡）
- **无法直接对比**：未测量实际 GPU 内存占用
- **理论分析**：MoE 57B 模型约 110GB（FP16），需要 TP=2 才能装载
- **分析**：MoE 的内存占用确实是瓶颈，单卡无法运行

### 3. 吞吐量对比
- **MoE / Dense 吞吐比**：1536.22 / 4402.86 = **0.35x**（MoE 慢 65%）
- **是否符合预期**：□ 是 ☑ 否（预期 0.5-1.5x，但实际更慢）
- **分析**：
  - MoE 显著慢于 Dense，可能原因：
    1. **TP 通信开销**：MoE 使用 TP=2，引入 All-Reduce 通信
    2. **MoE 调度开销**：Router + Expert 选择 + Token permutation
    3. **负载不均**：64 experts 在 batch=100 下可能分布不均
    4. **模型规模差异**：14B activated vs 7B，计算量仍是 2 倍

### 4. 延迟对比
- **TTFT**：MoE 3090.71ms vs Dense 1097.94ms（**MoE 慢 2.8x**）
- **TPOT**：MoE 26.62ms vs Dense 8.84ms（**MoE 慢 3.0x**）
- **分析**：
  - **TTFT 差异**：prefill 阶段，MoE 的 Router + Expert 调度开销明显
  - **TPOT 差异**：decode 阶段，即使单 token，MoE 仍需完整调度流程
  - **TP 影响**：TP=2 的通信延迟叠加到每个 token

## MoE 推理挑战验证

### 挑战 1: 内存墙
- [x] 验证：MoE 57B 模型需要 TP=2（两卡），单卡无法运行
- **影响**：部署成本高，需要多卡环境

### 挑战 2: 负载不均
- [ ] 待 Day 5 实验验证：不同 expert 的 token 分布
- **预期**：部分 expert 成为瓶颈
- **exp2 侧面验证**：batch=2048 时 FusedMoE kernel 达到 89.81 TFLOP/s，但端到端吞吐仍低

### 挑战 3: GPU 利用率
- **观察到的问题**：MoE 吞吐量仅为 Dense 的 35%
- **原因推测**：
  1. Token permutation 开销（按 expert 排序）
  2. Expert 计算负载不均导致串行化
  3. TP=2 引入的通信延迟
  4. Router 计算开销

## exp2 Kernel Profiling 补充分析

从 exp2 的 FusedMoE kernel profiling 结果：
- **Batch 16**: 2.22 TFLOP/s（GPU 严重空闲）
- **Batch 2048**: 89.81 TFLOP/s（提升 40 倍）

**关键发现**：
- MoE kernel 本身在大 batch 下能达到较高吞吐
- 但 exp1 端到端测试中，即使 batch=100，MoE 仍比 Dense 慢 3 倍
- **说明**：端到端性能瓶颈不只是 MoE 计算，还包括调度、通信、负载均衡等系统开销

## 结论

1. **MoE 优势场景**：
   - 大 batch、高并发场景（可以充分利用所有 expert）
   - 对参数量/质量要求高，但对延迟不敏感的场景
   - 理论上可以用更少的激活参数达到更好的效果

2. **MoE 劣势场景**（本实验观察）：
   - 小 batch、低延迟场景（expert 利用率低）
   - 需要 TP/EP 的场景（引入通信开销）
   - 相比同等激活参数的 Dense 模型，实际吞吐更低

3. **实验局限性**：
   - Dense 7B vs MoE 14B activated（模型规模不同）
   - Dense 单卡 vs MoE TP=2（配置不同）
   - 无法隔离 MoE 架构本身的影响 vs TP 通信的影响

4. **下一步探索**：
   - Day 3-4: 深入 FusedMoE kernel，理解性能瓶颈来源
   - Day 5: 分析 expert 路由分布，验证负载不均假设
   - 理想补充实验：对比 MoE 单卡 vs Dense 单卡（需要更小的 MoE 模型）
