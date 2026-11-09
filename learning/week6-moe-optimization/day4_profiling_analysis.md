# Day 4: FusedMoE Kernel Profiling 分析

## Experiment 2 结果

### 性能数据表
| Batch Size | Time (ms) | TFLOP/s | Tokens/s |
|-----------|----------|---------|----------|
| 16        | 13.56    | 2.22    | 1,180    |
| 32        | 15.82    | 3.80    | 2,023    |
| 64        | 16.02    | 7.51    | 3,994    |
| 128       | 16.65    | 14.44   | 7,687    |
| 256       | 17.03    | 28.24   | 15,030   |
| 512       | 22.74    | 42.31   | 22,515   |
| 1024      | 24.87    | 77.36   | 41,171   |
| 2048      | 42.85    | 89.81   | 47,794   |

### 关键指标
- **Peak TFLOP/s**: 89.81 (at batch_size=2048)
- **Saturation Point**: batch_size ≈ **未饱和**（性能在 2048 时仍在增长，需要更大 batch 测试）
- **GPU Utilization**: 28.8% (假设 A100 312 TFLOP/s FP16 峰值)
- **注**：实际硬件是 H800，FP16 峰值约 **1,979 TFLOP/s**（Tensor Core），实际利用率约 **4.5%**

## 分析

### 1. Batch Size 对性能的影响

#### 小 Batch (< 128)
- **观察**：TFLOP/s 从 2.22 (bs=16) 缓慢增长到 14.44 (bs=128)，增长 6.5x
- **原因**：
  1. GPU 严重未饱和：SM 利用率低，大量 SM 空闲
  2. Kernel launch 开销占比大：MoE 需要 permutation + 多次 GEMM，launch 开销显著
  3. 内存带宽未充分利用：数据传输量小，带宽闲置
- **TFLOP/s 相对峰值的比例**：2.5% - 16% (远低于峰值)
- **推理场景影响**：在线推理（小 batch）场景下 MoE 性能极差

#### 中等 Batch (128-512)
- **观察**：TFLOP/s 从 14.44 快速增长到 42.31，增长 2.9x，增速加快
- **原因**：
  1. GPU 开始饱和：更多 SM 被激活
  2. Token permutation 效果显现：相同 expert 的 token 聚集后可以批量计算
  3. 内存访问模式改善：更大的连续内存访问
- **特点**：性能增长斜率最大的区间，是吞吐/延迟的平衡点

#### 大 Batch (> 512)
- **观察**：TFLOP/s 从 42.31 (bs=512) 继续增长到 89.81 (bs=2048)，但增速放缓
- **原因**：
  1. 延迟增加明显：Time 从 22.74ms 跳升到 42.85ms (+88%)
  2. 内存带宽压力增大：大量 expert 权重需要反复读取
  3. Expert 负载不均的影响放大：热门 expert 成为瓶颈
- **关键发现**：Tokens/s 在 2048 时为 47,794，但延迟已达 42.85ms，不适合延迟敏感场景

### 2. 与 Dense FFN 的对比

#### 理论对比
```
Dense FFN FLOP per token:
  = 2 * (hidden_size * intermediate_size)    # up_proj
  + 2 * (intermediate_size * hidden_size)    # down_proj
  = 4 * 4096 * 14336
  ≈ 235 MFLOP

MoE FLOP per token (top_k=8, num_experts=64):
  = top_k * Dense_FLOP
  = 8 * 235 MFLOP
  = 1.88 GFLOP
  
但实际参数量：MoE = 64 * Dense = 64x 内存占用
```

#### 实际性能对比
- **MoE 实际 TFLOP/s**: 89.81 (at batch=2048)
- **Dense FFN 理论 TFLOP/s**（假设 H800，batch=2048）: ~300-400 TFLOP/s（基于 GEMM 优化实现）
- **MoE / Dense 比值**: ~0.22-0.30 (MoE 仅达到 Dense 性能的 22-30%)
- **结论**：
  1. **计算效率低**：MoE 虽然计算量是 Dense 的 8x (top_k=8)，但实际性能远低于 8 个连续 Dense FFN
  2. **内存墙显著**：64 个 expert 权重 (64x 内存占用) 导致频繁的内存访问，带宽成为瓶颈
  3. **负载不均代价**：Token 分布不均导致部分 expert 过载，无法充分并行
  4. **Permutation 开销**：Token 重排序引入额外计算和内存移动
  5. **部署权衡**：MoE 需要更大内存换取模型容量，但单位内存的计算效率较低

### 3. 性能瓶颈识别

#### Token Permutation 开销
- **估算方法**：
  1. Permutation 包括 argsort（按 expert_id 排序）+ gather/scatter 操作
  2. 理论开销：O(num_tokens * log(num_tokens)) 排序 + O(num_tokens) 重排
  3. 对比纯 GEMM：permutation 引入额外的全局内存访问
- **占比估算**：
  - 小 batch (16-64)：permutation 开销可能占 10-20%（相对开销大）
  - 大 batch (1024+)：permutation 开销降至 5% 以下（被 GEMM 淹没）
- **优化方向**：使用 GPU 高效排序算法（如 CUB radix sort）

#### Expert 负载不均
- **理想情况**：每个 expert 接收 (num_tokens * top_k / num_experts) 个 token
  - 示例：2048 tokens, top_k=8, 64 experts → 每个 expert 理想接收 256 tokens
- **实际情况**：
  - Router 学习到的分布通常不均：某些 expert 是"通用专家"，被频繁选中
  - 热门 expert 可能接收 500+ tokens，冷门 expert 仅 50-100 tokens
  - **影响**：最慢的 expert 决定整体延迟（木桶效应）
- **Week 6 Day 5 实验目标**：测量实际的 expert 路由分布

#### 内存带宽限制
- **H800 内存带宽**: 3,352 GB/s (HBM3)
- **是否受限**：☑ 是
- **判断依据**：
  1. **权重大小**：64 experts × (2×14336×4096 + 14336×4096) × 2 bytes ≈ 22 GB
  2. **每次 forward 需要读取的权重**：虽然只激活 top_k=8，但需要根据 token 分布动态读取，缓存命中率低
  3. **计算强度（FLOPs/Byte）分析**：
     - FLOPs per token: 1.88 GFLOP
     - 内存访问 per token（粗略）：读 8 个 expert 权重 ≈ 8 × (2×14336×4096 + 14336×4096) × 2 / num_tokens_per_expert bytes
     - 计算强度低于 Dense GEMM（通常 100+ FLOPs/Byte），接近 memory-bound
  4. **GPU 利用率低（4.5%）**：说明计算单元空闲，等待数据传输
- **结论**：MoE 是典型的 **memory-bound** workload，不是 compute-bound

## 与 Week 5 优化的对比

### CUDA Graph 对 MoE 的影响
- **预期**：MoE 的动态路由可能不兼容 CUDA Graph
- **Week 5 结论**：CUDA Graph 降低 decode TPOT 36.2%
- **MoE 情况**：
  1. **兼容性问题**：Router logits → top-k 选择 → expert_ids 是**动态的**，每个 token 的路由在运行时决定
  2. **CUDA Graph 要求**：固定的计算图（kernel 序列、参数、内存地址）
  3. **可能方案**：
     - ✗ 完全不兼容：如果 expert_ids 变化导致 kernel dispatch 路径不同
     - ✓ 部分兼容：如果 FusedMoE kernel 内部处理动态路由，graph 可以 capture
  4. **实际行为**（需 Day 7 验证）：
     - vLLM 可能对 MoE 层**禁用 CUDA Graph**，或
     - FusedMoE kernel 设计为 graph-friendly（固定 kernel，内部动态分支）
  5. **性能影响预估**：MoE decode 可能**无法享受 CUDA Graph 优化**，TPOT 提升有限

### Chunked Prefill 对 MoE 的影响
- **预期**：长 prefill 时，chunk 可以分摊 MoE 的负载不均
- **分析**：
  1. **问题**：单次大 batch prefill 时，expert 负载不均导致延迟长尾
  2. **Chunked Prefill 的帮助**：
     - 将 2048 tokens 拆成 4 × 512，分 4 次处理
     - 每次 chunk 的 expert 分布不同，可以**均摊**热门 expert 的压力
     - 降低单次 forward 的峰值延迟
  3. **代价**：增加总处理时间（4 次 kernel launch 开销）
  4. **适用场景**：在线服务中，用户体验对 P99 延迟敏感，chunked prefill 可以平滑延迟尖刺
- **待验证**：Day 7 实验测量 chunked vs non-chunked 的 P99 TTFT

## 结论

### FusedMoE Kernel 的优势
1. **统一的 kernel 接口**：将 64 个 expert 的计算融合为单次 kernel launch，避免了 64×2=128 次独立 GEMM 调用
2. **Token permutation 优化**：将相同 expert 的 token 聚集后批量计算，提升 GEMM 效率
3. **内存访问优化**：通过 shared memory 和 register blocking 减少 HBM 访问
4. **可扩展性**：支持任意数量的 expert 和 top_k，适应不同 MoE 架构

### FusedMoE Kernel 的局限
1. **Memory-bound 特性**：64 个 expert 权重导致内存压力大，GPU 利用率低（4.5%）
2. **小 batch 性能差**：batch < 128 时严重低效（TFLOP/s < 15），不适合在线推理
3. **负载不均敏感**：热门 expert 成为瓶颈，限制并行效率
4. **CUDA Graph 兼容性存疑**：动态路由可能无法享受 graph 优化
5. **相对 Dense FFN 效率低**：仅达到 Dense 性能的 22-30%，单位内存的计算效率不佳

### 生产部署建议
- **最优 batch size 范围**：512-1024
  - 理由：在吞吐（42-77 TFLOP/s）和延迟（22-25ms）之间取得平衡
  - batch < 256：吞吐过低，GPU 利用率 < 30%
  - batch > 1024：延迟激增，不适合在线场景
- **是否适合低延迟场景**：☑ 否
  - 原因：小 batch 性能差 + 负载不均导致延迟长尾
  - 低延迟场景建议：使用 Dense 模型或更小规模的 MoE（如 8 experts）
- **适合场景**：
  1. **离线批处理**：大 batch (1024+) 处理大量文本，追求吞吐而非延迟
  2. **中等 QPS 服务**：可以积累 batch 到 512+ 再处理
  3. **内存受限但需要大模型容量**：MoE 用内存换容量，适合无法部署等效 Dense 模型的场景
  4. **异构推理**：结合 EP (Expert Parallelism) 在多卡上分散 expert，缓解单卡内存压力（Day 6 学习）

## 遗留问题（Day 5-7 解决）

1. Expert 路由分布是否均匀？（Day 5 实验）
2. EP 能否解决负载不均问题？（Day 6）
3. 与其他优化（CP, Prefix Cache）的组合效果？（Day 7）
