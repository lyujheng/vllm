# Day 3: FusedMoE Kernel 实现分析

## 1. 函数入口定位

### fused_experts() 函数
- **位置**：`vllm/model_executor/layers/fused_moe/fused_moe.py:1587`
- **作用**：MoE 计算的高层接口，调用底层 C++/Triton kernel

### 函数签名
```python
def fused_experts(
    hidden_states: torch.Tensor,     # [num_tokens, hidden_size]
    w1: torch.Tensor,                # [num_experts, 2*intermediate_size, hidden_size]
    w2: torch.Tensor,                # [num_experts, hidden_size, intermediate_size]
    topk_weights: torch.Tensor,      # [num_tokens, top_k] router 输出权重
    topk_ids: torch.Tensor,          # [num_tokens, top_k] 选中的 expert ID
    ...
) -> torch.Tensor:
```

### 核心调用链
```
fused_experts() (Python wrapper)
  ↓
torch.ops.vllm.fused_experts (注册的 C++ op)
  ↓
fused_experts_impl() (实际实现)
  ↓
invoke_fused_moe_kernel() (调度到具体 backend)
```

## 2. Token Permutation 逻辑

### 为什么需要 Permutation？
```
原始 token 分布（随机）：
Token 0 → Expert 3, 7
Token 1 → Expert 1, 5
Token 2 → Expert 3, 9
...

问题：不同 expert 的 token 分散在内存中，无法高效 batch 计算

解决：按 expert 重排
Expert 1: [Token 1, ...]
Expert 3: [Token 0, Token 2, ...]
Expert 5: [Token 1, ...]
...

现在每个 expert 的 token 连续，可以批量 GEMM
```

### 关键函数调用链
```python
_prepare_expert_assignment()  # Line 1538
  ↓
moe_align_block_size()       # Line 11 in moe_align_block_size.py
  ↓
ops.moe_align_block_size()   # C++ custom op
```

### moe_align_block_size() 详解
**位置**：`vllm/model_executor/layers/fused_moe/moe_align_block_size.py:11`

**输入**：
- `topk_ids`: [num_tokens, top_k] - 每个 token 选中的 expert ID
- `block_size`: 通常是 64，用于对齐 Triton kernel 的 tile size
- `num_experts`: expert 总数

**输出**：
- `sorted_token_ids`: 按 expert 排序后的 token 索引
- `expert_ids`: 每个 block 对应的 expert ID
- `num_tokens_post_padded`: padding 后的总 token 数

### 工作流程示例
```
输入: topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]
      block_size = 4, num_experts = 4

Step 1: 展平 topk_ids
  [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]  (12 tokens)

Step 2: 统计每个 expert 的 token 数
  Expert 1: 3 tokens
  Expert 2: 3 tokens
  Expert 3: 3 tokens
  Expert 4: 3 tokens

Step 3: Padding 到 block_size 的倍数
  每个 expert 需要 pad 1 个 token (3 → 4)
  添加 padding token [12, 12, 12, 12]

Step 4: 按 expert 排序
  sorted_token_ids = [3, 6, 9, 12,     # Expert 1
                      0, 4, 10, 12,    # Expert 2
                      1, 7, 11, 12,    # Expert 3
                      2, 5, 8, 12]     # Expert 4
  
  expert_ids = [1, 2, 3, 4]  (每个 block 对应的 expert)
```

### 为什么需要 Padding？
- Triton kernel 使用 block-wise GEMM
- 每个 block 处理 `block_size` 个 token
- Padding 确保每个 expert 的 token 数是 block_size 的倍数
- Padding token（索引 >= num_tokens）在计算中被忽略

## 3. FusedMoE Kernel 数据流

### 完整调用链
```
fused_experts() (Line 1587)
  ↓
torch.ops.vllm.fused_experts (注册的 C++ op)
  ↓
fused_experts_impl() (Line 1650)
  ↓
_prepare_expert_assignment() (Line 1538)
  ├─ 小 batch：直接使用 topk_ids.view(-1)
  └─ 大 batch/EP：调用 moe_align_block_size() 排序和对齐
  ↓
invoke_fused_moe_triton_kernel() (Line 763)
  ↓
fused_moe_kernel[grid](...) (Triton JIT kernel, Line 64)
```

### invoke_fused_moe_triton_kernel() 详解
**位置**：`vllm/model_executor/layers/fused_moe/fused_moe.py:763`

**关键参数**：
- `A`: hidden_states [num_tokens, hidden_size]
- `B`: expert 权重 (w1 或 w2)
- `C`: 输出 tensor
- `sorted_token_ids`: 排序后的 token 索引（可选）
- `expert_ids`: 每个 block 对应的 expert ID
- `topk_weights`: router 权重

**Kernel Grid 计算**：
```python
grid = lambda META: (
    triton.cdiv(EM, META["BLOCK_SIZE_M"])  # token 维度分块
    * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),  # hidden 维度分块
)
```
- `BLOCK_SIZE_M`: 通常 64，token 方向的 tile size
- `BLOCK_SIZE_N`: 通常 128，hidden 方向的 tile size
- 每个 thread block 计算一个 [M, N] 的 tile

### 数据流图
```
输入: hidden_states [N, D]
      topk_ids [N, K]
      topk_weights [N, K]
      
Step 1: Token Permutation (_prepare_expert_assignment)
  ┌─────────────────────────────────────┐
  │ moe_align_block_size()              │
  │  - 统计每个 expert 的 token 数      │
  │  - Padding 到 block_size 的倍数     │
  │  - 按 expert 排序 token             │
  └─────────────────────────────────────┘
  输出: sorted_token_ids, expert_ids
  
Step 2: Expert Computation (Triton Kernel)
  ┌─────────────────────────────────────┐
  │ fused_moe_kernel (Triton)           │
  │                                     │
  │ For each expert block:              │
  │   1. Gate projection (A @ w1)       │
  │      - 使用 sorted_token_ids        │
  │      - 使用 expert_ids 找权重       │
  │   2. Activation (SiLU)              │
  │   3. Down projection (act @ w2)     │
  │   4. 乘以 topk_weights (可选)       │
  └─────────────────────────────────────┘
  
Step 3: Unpermute & Aggregate
  - 恢复原始 token 顺序
  - 如果一个 token 选了多个 expert，加权求和
  
输出: [N, D]
```

### Triton Kernel 优势
1. **Fused Computation**：Gate + Activation + Down 融合在一个 kernel
2. **Block-wise GEMM**：充分利用 Tensor Core
3. **Shared Memory**：减少 global memory 访问
4. **Expert Batching**：相同 expert 的 token 批量计算

## 4. 性能关键点

### 优势
1. **Fused Kernel**：所有 expert 的 GEMM 融合，减少 kernel launch 开销
2. **Triton 优化**：充分利用 Tensor Core 和 shared memory
3. **Batched GEMM**：每个 expert 的 token 批量计算
4. **Block-wise 计算**：tile size 优化，适配 GPU 架构

### 瓶颈来源

#### 1. Token Permutation 开销
- **排序成本**：O(N*K*log(N*K))，N=tokens, K=top_k
- **内存访问**：需要重新排列 hidden_states
- **小 batch 时**：排序开销占比大

#### 2. 负载不均
```
理想情况（均匀分布）：
Expert 0: 100 tokens
Expert 1: 100 tokens
...
Expert 63: 100 tokens
→ 所有 expert 并行，GPU 满载

实际情况（可能不均）：
Expert 0: 200 tokens
Expert 1: 5 tokens
...
Expert 63: 50 tokens
→ 部分 expert 空闲，GPU 利用率低
```

#### 3. 小 Batch 问题（exp2 验证）
从 exp2 的 profiling 结果：
- **Batch 16**: 2.22 TFLOP/s
- **Batch 2048**: 89.81 TFLOP/s（40x 提升）

**原因分析**：
```
假设 64 experts, top_k=8, block_size=64

Batch 16:
  - 总 tokens: 16
  - 每个 token 激活 8 experts
  - 总激活: 16 * 8 = 128 expert-token pairs
  - 平均每个 expert: 128 / 64 = 2 tokens
  - 问题：2 tokens << 64 (block_size)
    → 大量 padding
    → GPU SM 利用率低
    → 无法填满 Tensor Core

Batch 2048:
  - 总 tokens: 2048
  - 总激活: 2048 * 8 = 16384 expert-token pairs
  - 平均每个 expert: 16384 / 64 = 256 tokens
  - 256 / 64 = 4 blocks per expert
    → 充分利用 GPU 并行度
    → Tensor Core 饱和
```

#### 4. TP 通信开销（exp1 观察）
- Exp1 中 MoE 使用 TP=2
- 每层 MoE 需要一次 All-Reduce
- 通信延迟叠加到延迟指标中

### exp2 结果解读

| Batch | Time(ms) | TFLOP/s | 分析 |
|-------|----------|---------|------|
| 16    | 13.56    | 2.22    | 严重 underutilized，平均每 expert 仅 2 tokens |
| 64    | 16.02    | 7.51    | 仍然不足，每 expert ~8 tokens |
| 256   | 17.03    | 28.24   | 开始改善，每 expert ~32 tokens |
| 1024  | 24.87    | 77.36   | 较好，每 expert ~128 tokens |
| 2048  | 42.85    | 89.81   | 最佳，每 expert ~256 tokens |

**关键发现**：
- 需要 **batch >= 256** 才能达到合理的 GPU 利用率
- Batch 越大，TFLOP/s 越高，但增长开始放缓（接近饱和）
- 小 batch 下，permutation 和 padding 开销占主导

### 与端到端性能对比（exp1 vs exp2）

**Exp1 端到端**（batch=100）：
- MoE: 1536 tok/s
- Dense: 4402 tok/s
- MoE 慢 65%

**Exp2 Kernel 单独**（batch=2048）：
- 89.81 TFLOP/s
- 说明 kernel 本身在大 batch 下能达到较高吞吐

**结论**：
- 端到端性能瓶颈 **不只是** MoE kernel 计算
- 还包括：
  1. Token permutation 开销
  2. Router 计算开销
  3. TP 通信开销（All-Reduce）
  4. 负载不均导致的等待
  5. Batch size 可能不够大（100 vs 2048）

## 学习总结

### ✅ 已完成
- [x] 定位 FusedMoE kernel 入口（`fused_experts`, Line 1587）
- [x] 理解 Token Permutation 实现（`moe_align_block_size`）
- [x] 找到 Triton kernel 调用位置（`invoke_fused_moe_triton_kernel`, Line 763）
- [x] 理解完整的数据流：Permutation → Triton Kernel → Unpermute
- [x] 分析小 batch 性能差的根本原因（结合 exp2 数据）
- [x] 理解 block-wise 计算和 padding 的必要性

### 核心认知

1. **MoE Kernel 三阶段**：
   - Permutation：按 expert 重排 token（开销）
   - Computation：Triton kernel 批量计算（核心）
   - Unpermute：恢复顺序并加权（开销）

2. **性能瓶颈本质**：
   - 小 batch 时，平均每个 expert 分到的 token 太少
   - 无法充分利用 GPU 并行度（SM、Tensor Core）
   - Padding 和 permutation 开销占比高

3. **优化方向**：
   - 增大 batch size（提高并发）
   - 使用 Expert Parallelism（Day 5-6 学习）
   - 负载均衡策略（router 优化）

### 🔄 下一步：Day 4

Day 4 的 Profiling 实验（exp2）已经完成，建议：
- 创建 `day4_profiling_analysis.md`，深入分析 exp2 结果
- 或直接进入 Day 5：Expert Routing 分析与 EP 理论
