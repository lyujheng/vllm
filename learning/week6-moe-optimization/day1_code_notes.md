# Day 1: MoE 代码结构笔记

## Qwen2MoE 模型架构

### 文件位置
- **主文件**：`vllm/model_executor/models/qwen2_moe.py`
- **FusedMoE 层**：`vllm/model_executor/layers/fused_moe/fused_moe.py`

### 关键类

#### Qwen2MoeDecoderLayer
- **行号**：___（使用 `grep -n "class Qwen2MoeDecoderLayer"` 查找）
- **self.self_attn**：标准 Attention 层
- **self.mlp**：`Qwen2MoeSparseMoeBlock` 实例
- **forward 流程**：
  ```python
  # 伪代码
  hidden_states = self.input_layernorm(hidden_states)
  hidden_states = self.self_attn(hidden_states, ...)
  
  # MoE 层
  hidden_states = self.post_attention_layernorm(hidden_states)
  hidden_states = self.mlp(hidden_states)  # 调用 MoE block
  ```

#### Qwen2MoeSparseMoeBlock
- **行号**：___
- **关键成员**：
  - `self.gate`：Router，计算 expert 选择 logits
  - `self.experts`：ModuleList，包含所有 expert（通常不直接用）
  - `self.num_experts`：Total expert 数量（如 64）
  - `self.top_k`：每个 token 激活的 expert 数（如 8）
  - `self.shared_expert`：共享 expert（某些模型有）

- **forward 流程**：
  ```python
  # 1. Router 计算
  router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]
  
  # 2. Top-K 选择
  topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
  topk_weights = F.softmax(topk_weights, dim=-1)
  
  # 3. 调用 fused_moe kernel
  output = fused_moe(
      hidden_states,
      w1=self.w13_weight,  # gate_up_proj 权重 [num_experts, 2*intermediate, hidden]
      w2=self.w2_weight,   # down_proj 权重 [num_experts, hidden, intermediate]
      topk_weights=topk_weights,
      topk_ids=topk_ids,
      ...
  )
  
  # 4. 可选：加上 shared expert 输出
  if self.shared_expert is not None:
      output += self.shared_expert(hidden_states)
  
  return output
  ```

### Expert 权重布局
- **w13_weight**（gate_up_proj）：
  - Shape: `[num_experts, 2 * intermediate_size, hidden_size]`
  - 包含 gate_proj 和 up_proj，按维度拼接
  
- **w2_weight**（down_proj）：
  - Shape: `[num_experts, hidden_size, intermediate_size]`

### 配置示例（Qwen2-57B-A14B-Instruct）
```python
{
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_experts": 64,
    "num_experts_per_tok": 8,  # top_k
    "shared_expert_intermediate_size": 2048,  # 如果有 shared expert
}
```

## 问题清单（后续回答）

### Day 1-2 问题
1. **Router 如何防止负载不均？**
   - A: ___（查找 auxiliary_loss 或 load_balancing_loss）

2. **Shared Expert 的作用是什么？**
   - A: ___（某些 token 可能不适合任何 specialist expert）

3. **Top-K 选择后，如果某些 expert 接收 0 个 token，会发生什么？**
   - A: ___

### Day 3-4 问题
1. **fused_moe() 如何组织所有 expert 的权重？**
   - A: ___（从 day3_kernel_dataflow.md 填写）

2. **w13_weight 中 gate 和 up 如何区分？**
   - A: 前半部分是 gate，后半部分是 up（通过 split）

3. **inplace=True 的作用？**
   - A: ___（内存优化，直接修改输入）

## 代码阅读任务清单

### 必读代码段
- [ ] `Qwen2MoeDecoderLayer.forward()` 完整流程
- [ ] `Qwen2MoeSparseMoeBlock.__init__()` 权重初始化
- [ ] `Qwen2MoeSparseMoeBlock.forward()` Router + fused_moe 调用
- [ ] `fused_moe()` 函数签名和文档字符串
- [ ] 模型配置文件（如果可用）：`config.json` 中的 MoE 参数

### 可选深入
- [ ] `Qwen2MoeModel.load_weights()` 权重加载逻辑
- [ ] TP 模式下 expert 权重的切分（如果支持 TP）
- [ ] Router 的 load balancing 相关代码

## 绘图任务

### MoE Layer 结构图
```
Input: hidden_states [num_tokens, hidden_size]
         ↓
    ┌─────────────┐
    │   Router    │  (Linear: hidden_size → num_experts)
    └─────────────┘
         ↓
    router_logits [num_tokens, num_experts]
         ↓
    ┌─────────────┐
    │   Top-K     │  (torch.topk)
    └─────────────┘
         ↓
    topk_ids, topk_weights [num_tokens, top_k]
         ↓
    ┌──────────────────────────────┐
    │      FusedMoE Kernel         │
    │  (All experts computed       │
    │   in single kernel launch)   │
    └──────────────────────────────┘
         ↓
    output [num_tokens, hidden_size]
         ↓
    (可选) + Shared Expert output
         ↓
    Final output
```

### Expert 权重布局图
```
w13_weight: [num_experts, 2*intermediate, hidden]
┌─────────────────────────────────────┐
│ Expert 0:                           │
│   gate_proj: [intermediate, hidden] │
│   up_proj:   [intermediate, hidden] │ ← 拼接
├─────────────────────────────────────┤
│ Expert 1:                           │
│   gate_proj + up_proj               │
├─────────────────────────────────────┤
│ ...                                 │
├─────────────────────────────────────┤
│ Expert 63:                          │
│   gate_proj + up_proj               │
└─────────────────────────────────────┘

w2_weight: [num_experts, hidden, intermediate]
┌─────────────────────────────────────┐
│ Expert 0: down_proj                 │
├─────────────────────────────────────┤
│ Expert 1: down_proj                 │
├─────────────────────────────────────┤
│ ...                                 │
└─────────────────────────────────────┘
```

## 与 Dense FFN 的对比

| 组件 | Dense FFN | MoE |
|-----|----------|-----|
| gate_proj | 1 个 Linear | num_experts 个 |
| up_proj | 1 个 Linear | num_experts 个 |
| down_proj | 1 个 Linear | num_experts 个 |
| Router | 无 | 1 个 Linear |
| 参数量 | 3 * hidden * intermediate | num_experts * 3 * hidden * intermediate + router |
| 计算量 / token | 固定 | 取决于 top_k |

## 学习验证

完成以下任务后，你应该能：
1. 在代码中找到 router 的定义和 forward 逻辑
2. 解释 topk_weights 和 topk_ids 的含义
3. 说明 fused_moe() 的输入和输出
4. 理解为什么需要将 gate_proj 和 up_proj 拼接（减少 kernel launch）
