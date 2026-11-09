# Day 3: FusedMoE Kernel 数据流分析

## Kernel 执行流程

### 输入数据结构
```python
# fused_moe() 函数签名（从代码中提取）
def fused_moe(
    hidden_states: Tensor,      # [num_tokens, hidden_size]
    w1: Tensor,                 # [num_experts, 2*intermediate_size, hidden_size]
    w2: Tensor,                 # [num_experts, hidden_size, intermediate_size]
    topk_weights: Tensor,       # [num_tokens, top_k] - router 权重
    topk_ids: Tensor,           # [num_tokens, top_k] - expert ID
    ...
) -> Tensor:                    # [num_tokens, hidden_size]
```

### 执行步骤（填写行号）

#### Step 1: Token Permutation
- **目的**：将相同 expert 的 token 聚集，实现批量 GEMM
- **代码位置**：`fused_moe.py` 行 ___
- **关键操作**：
  ```python
  # 伪代码
  expert_ids_flat = topk_ids.flatten()  # [num_tokens * top_k]
  sorted_indices = expert_ids_flat.argsort()  # 按 expert ID 排序
  sorted_tokens = hidden_states[sorted_indices]  # 重排 token
  ```

#### Step 2: Expert Computation
- **代码位置**：`fused_moe.py` 行 ___
- **Triton Kernel 调用位置**：行 ___
- **操作**：
  ```
  For each expert e:
    tokens_for_e = sorted_tokens[expert_e_start : expert_e_end]
    # GEMM 1: gate_up projection
    intermediate = tokens_for_e @ w1[e].T  # [n_tokens_e, 2*intermediate_size]
    # Activation (SwiGLU)
    gate, up = split(intermediate)
    activated = gate * silu(up)
    # GEMM 2: down projection
    output_e = activated @ w2[e].T  # [n_tokens_e, hidden_size]
  ```

#### Step 3: Result Unpermutation
- **代码位置**：`fused_moe.py` 行 ___
- **操作**：
  ```python
  # 将结果还原到原始 token 顺序
  output = scatter(expert_outputs, sorted_indices)
  # 应用 router 权重
  output = output * topk_weights
  ```

## 数据流图

```
Input: hidden_states [N, D]
         ↓
    ┌────────────────┐
    │ Router (外部)   │
    └────────────────┘
         ↓
    topk_ids, topk_weights
         ↓
┌──────────────────────┐
│ Step 1: Permutation  │  ← argsort by expert_id
│  Sort tokens by      │
│  expert assignment   │
└──────────────────────┘
         ↓
    sorted_tokens [N*K, D]
         ↓
┌──────────────────────────────────────┐
│ Step 2: Fused Expert Computation     │
│                                      │
│  Expert 0: [batch_0, D] @ w1[0]     │
│  Expert 1: [batch_1, D] @ w1[1]     │
│  ...                                 │
│  Expert E-1: [batch_E-1, D] @ w1[E-1]│
│                                      │
│  (所有 expert 在一个 Triton kernel)  │
└──────────────────────────────────────┘
         ↓
    expert_outputs [N*K, D]
         ↓
┌──────────────────────┐
│ Step 3: Unpermute    │  ← scatter back
│  + Weight by router  │
└──────────────────────┘
         ↓
    output [N, D]
```

## 关键优化点

### 1. Fused Kernel 的优势
- **传统方案**：为每个 expert 单独 launch kernel → E 次 kernel launch 开销
- **Fused 方案**：所有 expert 在一个 kernel → 1 次 kernel launch
- **代价**：需要 token permutation（sort 开销）

### 2. Token Permutation 的必要性
- **目的**：将相同 expert 的 token 聚集 → GEMM 可以批量计算
- **Without permutation**：每个 token 需单独调用对应 expert → 无法利用矩阵乘法
- **With permutation**：相同 expert 的 token 组成 mini-batch → 高效 GEMM

### 3. 性能瓶颈分析
- [ ] **Sort 开销**：argsort 的复杂度 O(N*K * log(N*K))
- [ ] **负载不均**：如果某个 expert 接收过多 token → 成为瓶颈
- [ ] **内存带宽**：大量 token shuffle 操作

## 问题记录

### Q1: fused_moe() 如何避免为每个 expert 单独 kernel launch？
**A**: ___（Day 4 通过阅读 Triton kernel 代码回答）

### Q2: Token permutation 的实际开销占比多少？
**A**: ___（通过 exp2 profiling 结果推测）

### Q3: Expert Parallelism 时，all-to-all 通信发生在哪一步？
**A**: ___（Day 5 学习）

## 代码阅读清单（完成后打勾）

- [ ] 找到 `fused_moe()` 函数定义（行号：___）
- [ ] 找到 token permutation 代码段（行号：___）
- [ ] 找到 Triton kernel 调用（行号：___）
- [ ] 找到 unpermutation 代码段（行号：___）
- [ ] 阅读 Triton kernel 的 `@triton.jit` 定义（如果在同文件）
- [ ] 理解 `inplace` 参数的作用
