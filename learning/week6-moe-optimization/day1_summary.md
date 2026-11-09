# Week 6 Day 1 学习总结

**学习时间**：2024-XX-XX  
**学习内容**：MoE 架构原理与代码阅读  
**学习时长**：约 4 小时

---

## 一、核心概念理解

### 1.1 MoE (Mixture of Experts) 基本原理

**定义**：将大模型的 FFN 层替换为多个专家（Expert）网络，每个 token 只激活其中部分 expert。

**核心优势**：
- **参数-计算解耦**：57B 参数，只激活 14B（稀疏激活）
- **专精化**：不同 expert 学习不同模式（如代码、数学、对话）
- **可扩展性**：增加 expert 数量提升模型容量，计算量不成比例增长

**关键组件**：
1. **Router (Gate)**：为每个 token 计算 expert 选择概率
2. **Experts**：多个并行的 FFN 网络
3. **Top-K Selection**：每个 token 选择 k 个最相关的 expert

---

### 1.2 Qwen2-57B-A14B 模型结构

#### 整体架构
```
输入 → Embedding → 28 × DecoderLayer → RMSNorm → LM Head → 输出
```

#### 每个 DecoderLayer
```
Input
  ↓
Attention Layer (标准 Transformer Attention)
  ↓ (残差连接)
MoE Layer (Routed + Shared 并行)
  ↓ (残差连接)
Output
```

#### MoE Layer 详细结构
```
hidden_states (4096 维)
    ├─→ Routed Path
    │     ├─ Router (4096 → 64) 
    │     ├─ Top-K Selection (选 8 个)
    │     └─ FusedMoE (64 experts, 每个 4096→14336→4096)
    │
    └─→ Shared Path
          └─ Shared Expert (4096→2048→4096, 所有 token 过)

final_output = routed_output + shared_output
```

**关键参数**：
- 总层数：28 layers
- Expert 数量：64
- Top-K：8（每个 token 激活 8 个 expert）
- Routed Expert 中间层：14336
- Shared Expert 中间层：2048
- 激活率：8/64 = 12.5%

---

## 二、代码阅读笔记

### 2.1 关键文件和行号

| 组件 | 文件路径 | 关键行号 |
|-----|---------|---------|
| MoE Block | `vllm/model_executor/models/qwen2_moe.py` | 121-189 |
| Decoder Layer | `vllm/model_executor/models/qwen2_moe.py` | 281-340 |
| FFN (MLP) | `vllm/model_executor/models/qwen2_moe.py` | 70-113 |
| FusedMoE Factory | `vllm/model_executor/layers/fused_moe/layer.py` | 99+ |

### 2.2 Qwen2MoeSparseMoeBlock 代码解析

**初始化（行 121-175）**：
```python
class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(...):
        # Router: 为每个 token 计算 64 个 expert 的选择概率
        self.gate = ReplicatedLinear(
            hidden_size,      # 4096
            num_experts,      # 64
        )
        
        # Shared Expert Gate: 控制 shared expert 输出权重
        self.shared_expert_gate = ReplicatedLinear(
            hidden_size, 1
        )
        
        # Shared Expert: 所有 token 无条件执行
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                intermediate_size=2048
            )
        
        # Routed Experts: 64 个 expert，通过 FusedMoE 执行
        self.experts = FusedMoEFactory(
            num_experts=64,
            top_k=8,
            hidden_size=4096,
            intermediate_size=14336,
            shared_experts=self.shared_expert,
        )
```

**前向传播（行 177-189）**：
```python
def forward(self, hidden_states):
    # 1. Router 计算
    router_logits, _ = self.gate(hidden_states)
    
    # 2. FusedMoE 执行（内部处理 Top-K + Expert 计算 + Shared Expert）
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits
    )
    
    return final_hidden_states
```

**关键观察**：
- Shared Expert 通过 `FusedMoEFactory` 的 `shared_experts` 参数传入
- Router 只计算 logits，具体的 Top-K 选择在 `self.experts` 内部
- 前向传播逻辑简洁，复杂度封装在 FusedMoE 内部

---

### 2.3 FFN 结构解析

**标准 FFN (Qwen2MoeMLP)**：
```python
# 行 86-102
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size=4096,
    [intermediate_size] * 2,  # [14336, 14336]
)
self.down_proj = RowParallelLinear(
    intermediate_size=14336,
    hidden_size=4096,
)

# 前向传播（行 111-113）
gate_up, _ = self.gate_up_proj(x)        # [N, 28672]
out = self.act_fn(gate_up)                # SiluAndMul: split + silu + mul
out, _ = self.down_proj(out)              # [N, 4096]
```

**SwiGLU 激活**：
- `gate_up_proj` 合并了 `gate_proj` 和 `up_proj`（减少 kernel launch）
- `SiluAndMul()` 融合了 `split + silu + multiply` 操作
- 等价于：`output = down(gate(x) ⊙ silu(up(x)))`

---

### 2.4 层级选择逻辑

**哪些层使用 MoE？（行 313-318）**
```python
layer_idx = extract_layer_index(prefix)
mlp_only_layers = getattr(config, "mlp_only_layers", [])

if (layer_idx not in mlp_only_layers) and \
   (config.num_experts > 0 and 
    (layer_idx + 1) % config.decoder_sparse_step == 0):
    # 使用 MoE
    self.mlp = Qwen2MoeSparseMoeBlock(...)
else:
    # 使用 Dense FFN
    self.mlp = Qwen2MoeMLP(...)
```

**Qwen2-57B-A14B 的配置**：
- `decoder_sparse_step = 1` → 每层都是 MoE
- `mlp_only_layers = []` → 没有强制 Dense 的层
- **结论**：所有 28 层都使用 MoE

---

## 三、核心问题理解

### Q1: Router 如何防止负载不均？

**答案**：vLLM 在推理阶段**不做负载均衡**。

**详细解释**：
- **训练时**：模型训练时会加入 `load_balancing_loss`（辅助损失），鼓励 Router 均匀分配 token 到各 expert
- **推理时**：Router 权重已固定，vLLM 直接使用训练好的 Router，不再调整
- **依赖**：依赖模型训练阶段已经学到的负载均衡性

**如果负载不均的后果**：
- 某些 expert 过载 → 成为性能瓶颈
- 某些 expert 空闲 → GPU 资源浪费
- **解决方案**：Week 8 会学习 EPLB (Expert-Parallel Load Balancing) 动态重分配

---

### Q2: Shared Expert 的作用是什么？

**答案**：提取**所有 token 都需要的通用特征**。

**设计动机**：
- Routed Expert 通过 Top-K 选择，每个 token 只能访问部分 expert
- 通用知识（语法、常见搭配）如果靠 Routed Expert 学习，会导致各 expert 冗余
- Shared Expert 对所有 token 无条件执行，专门捕获通用模式

**类比**：
- Routed Expert = 专科医生（64 个，每个病人看 8 个）
- Shared Expert = 全科医生（1 个，每个病人都过）

**实现细节**：
- Shared Expert 的 intermediate_size 更小（2048 vs 14336）
- 有独立的 `shared_expert_gate` 控制输出权重
- 与 Routed Expert 的输出**相加**（并行关系）

**并非所有 MoE 都有 Shared Expert**：
- Qwen2-MoE：有
- DeepSeek-V2：有
- Mixtral-8x7B：无

---

### Q3: Top-K 选择后，某些 expert 接收 0 个 token 会怎样？

**答案**：该 expert 被**跳过，不执行计算**。

**业务场景**：
1. **小 Batch**：batch_size=16, top_k=8 → 总选择 128 次，但有 64 个 expert → 必然有 expert 空闲
2. **Router 偏好**：某些任务（如代码生成）倾向选择特定 expert，其他 expert 很少被选
3. **Decode 阶段**：单 token 生成时，1 token × 8 = 只激活 8 expert，其余 56 个空闲
4. **特定领域输入**：输入类型单一（如全是数学公式），只需要少数专精 expert

**FusedMoE Kernel 的处理**：
```python
for expert_id in range(num_experts):
    token_count = count_tokens_for_expert(expert_id)
    if token_count == 0:
        continue  # 跳过，不启动 thread block
    
    # 只为有 token 的 expert 执行计算
    process_expert(expert_id, tokens)
```

**影响**：
- ✅ 节省计算（不浪费 FLOPs）
- ❌ 显存仍占用（expert 权重已加载）
- ❌ GPU 利用率降低（并行度减少）

---

## 四、深入理解

### 4.1 FusedMoE 的含义

**Fused = 融合**，将多个 expert 的计算合并到一个 kernel。

**对比**：

| | Naive MoE | FusedMoE |
|---|-----------|----------|
| Kernel Launches | 64 expert × 3 ops = 192 次 | **1 次** |
| Token 处理 | 逐个 expert 串行 | 所有 expert 并行 |
| GPU 利用率 | 低（小 workload） | 高（大 workload） |
| 延迟 | ~100ms | ~15ms (6.7x 提升) |

**核心优化**：
1. 减少 kernel launch overhead
2. 提高并行度（64 expert 映射到多个 thread block）
3. 更好的 cache locality

---

### 4.2 Tensor Parallelism（TP）

**Column + Row 组合**是 TP 的标准模式。

**MergedColumnParallelLinear（gate_up_proj）**：
- 权重按**列切分**：`[4096, 28672]` → GPU0: `[4096, 14336]`, GPU1: `[4096, 14336]`
- 输入广播到所有 GPU
- 输出拼接：`[N, 14336] + [N, 14336] = [N, 28672]`
- **无通信**

**RowParallelLinear（down_proj）**：
- 权重按**行切分**：`[14336, 4096]` → GPU0: `[7168, 4096]`, GPU1: `[7168, 4096]`
- 输入已分块（来自上层 Column 输出）
- 输出 All-Reduce 求和：`out0 + out1 = [N, 4096]`
- **1 次 All-Reduce**

**为什么用 Row？**
- 自然衔接上层 Column 输出
- 减少通信（只需 1 次 All-Reduce，vs 多次 All-Gather）
- 数学等价：`x @ W = x0 @ W0 + x1 @ W1`

---

### 4.3 Expert Parallelism（EP）

**EP = 将不同 expert 分配到不同 GPU**。

**与 TP 的区别**：
- **TP**：单个 expert 的权重切分到多个 GPU
- **EP**：每个 GPU 持有不同 expert 的完整权重

**EP 下 FusedMoE 的执行**：
1. **Router 计算**：所有 GPU 独立计算（Router 权重 Replicated）
2. **All-to-All 通信**：将 token 发送到持有对应 expert 的 GPU
3. **FusedMoE 执行**：每个 GPU 正常执行 FusedMoE（但只处理自己的 expert）
4. **All-to-All 返回**：将结果发回原 GPU

**EP 的优势**：
- 显存需求降低（每个 GPU 只存部分 expert）
- 负载更局部化（某 expert 空闲只影响一个 GPU）

**EP 的代价**：
- 两次 All-to-All 通信（发送 + 接收）

**Day 5-6 会深入学习 EP**。

---

### 4.4 MoE 并行配置

**代码入口**（`fused_moe/layer.py:222-229`）：
```python
moe_parallel_config = make_parallel_config(
    tp_size=tp_size,        # Tensor Parallel
    dp_size=dp_size,        # Data Parallel (Attention 层)
    pcp_size=pcp_size,      # Expert Parallel
    is_sequence_parallel=is_sequence_parallel,
)
```

**配置示例（8 GPU）**：

| 配置 | 含义 |
|-----|------|
| `tp_size=2, pcp_size=4` | 每个 expert 权重 TP 切分到 2 GPU，4 组 expert |
| `tp_size=1, pcp_size=8` | 纯 EP，8 个 GPU 各持有不同 expert |
| `dp_size=2, pcp_size=4` | Attention DP 并行，MoE EP 并行 |

---

## 五、学习收获

### 已掌握的概念

1. ✅ MoE 的基本原理（Router + Expert + Top-K）
2. ✅ Qwen2-57B-A14B 的完整架构
3. ✅ Routed Expert + Shared Expert 的并行关系
4. ✅ FusedMoE 的融合优化原理
5. ✅ FFN 的 SwiGLU 结构
6. ✅ Tensor Parallelism 的 Column + Row 组合
7. ✅ Expert Parallelism 的基本概念
8. ✅ 负载不均的场景和影响

### 待深入的内容（后续 Day）

- **Day 2**：MoE vs Dense 性能对比实验
- **Day 3-4**：FusedMoE Kernel 的 Triton 实现
- **Day 5-6**：Expert Parallelism 和 All-to-All 通信
- **Day 7**：部署调优和性能优化

---

## 六、下一步计划

### Day 2 学习目标
运行 `exp1_moe_vs_dense.sh` 和 `exp2_profile_moe.py`，验证：
1. MoE 的内存占用 vs 计算量权衡
2. 不同 batch size 下的 expert 负载分布
3. FusedMoE 的性能特征

### 准备工作
- [x] 模型已下载：`/data/model/Qwen2-57B-A14B-Instruct`
- [ ] 确认 GPU 数量和显存：`nvidia-smi`
- [ ] 检查实验脚本：`ls learning/week6-moe-optimization/exp*.sh`

---

## 七、参考资料

### 代码位置
- Qwen2-MoE 模型：`vllm/model_executor/models/qwen2_moe.py`
- FusedMoE 实现：`vllm/model_executor/layers/fused_moe/`
- 模型结构文档：`learning/week6-moe-optimization/MODEL_STRUCTURE.md`

### 论文（可选阅读）
- Switch Transformers (Google, 2021) - MoE 在 Transformer 的现代应用
- GShard (Google, 2020) - Expert Parallelism 提出
- Mixtral of Experts (Mistral AI, 2024) - 实际部署经验

---

**Day 1 完成标志**：✅ 理解了 MoE 架构原理，准备进入实验验证阶段。
