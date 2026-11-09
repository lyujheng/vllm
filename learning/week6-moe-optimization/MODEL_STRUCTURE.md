# Qwen2-57B-A14B-Instruct 模型结构解析

## 一、整体架构

```
Qwen2MoeForCausalLM
├── model (Qwen2MoeModel)
│   ├── embed_tokens (词嵌入: vocab_size → hidden_size=4096)
│   │
│   ├── layers (28 个 Qwen2MoeDecoderLayer)
│   │   │
│   │   ├── Layer 0
│   │   │   ├── input_layernorm (RMSNorm)
│   │   │   ├── self_attn (Qwen2MoeAttention)
│   │   │   │   ├── q_proj (4096 → 4096, 32 heads)
│   │   │   │   ├── k_proj (4096 → 4096, 32 kv_heads)
│   │   │   │   ├── v_proj (4096 → 4096, 32 kv_heads)
│   │   │   │   ├── RoPE (旋转位置编码)
│   │   │   │   ├── Attention Computation (+ KV Cache)
│   │   │   │   └── o_proj (4096 → 4096)
│   │   │   ├── post_attention_layernorm (RMSNorm)
│   │   │   └── mlp (Qwen2MoeSparseMoeBlock)      ← MoE 层
│   │   │       ├── gate (Router: 4096 → 64)
│   │   │       ├── experts (FusedMoE: 64 experts, top-8)
│   │   │       │   ├── Expert 0: FFN (4096 → 14336 → 4096)
│   │   │       │   ├── Expert 1: FFN
│   │   │       │   ├── ...
│   │   │       │   └── Expert 63: FFN
│   │   │       ├── shared_expert (FFN: 4096 → 2048 → 4096)
│   │   │       └── shared_expert_gate (4096 → 1)
│   │   │
│   │   ├── Layer 1 (同上)
│   │   ├── ...
│   │   └── Layer 27
│   │
│   └── norm (最终 RMSNorm)
│
└── lm_head (hidden_size → vocab_size)
```

## 二、Qwen2MoeDecoderLayer 结构（核心）

每个 Decoder Layer 包含：

```
Input: hidden_states [num_tokens, hidden_size=4096]
   ↓
┌──────────────────────────────────────┐
│  1. Self-Attention                   │
│     (标准 Transformer Attention)      │
│     - Q, K, V projections            │
│     - Multi-head attention           │
│     - Output projection              │
└──────────────────────────────────────┘
   ↓ (残差连接)
┌──────────────────────────────────────────────────────────┐
│  2. MoE Layer (Qwen2MoeSparseMoeBlock)  ← Week 6 重点   │
│                                                          │
│  hidden_states ─────────────┬──────────────────────┐     │
│                             │                      │     │
│                             ↓                      ↓     │
│               ┌─── Routed Path ───┐   ┌─ Shared Path ─┐ │
│               │                   │   │                │ │
│               │  Router (gate)    │   │ Shared Expert  │ │
│               │  Linear: 4096→64  │   │ FFN: 4096→2048 │ │
│               │       ↓           │   │    →4096       │ │
│               │  Top-K (k=8)      │   │ (所有 token    │ │
│               │       ↓           │   │  无条件执行)   │ │
│               │  FusedMoE Kernel  │   │                │ │
│               │  64 experts       │   │ * shared_expert│ │
│               │  每个: FFN        │   │   _gate 控制   │ │
│               │  4096→14336→4096  │   │   输出权重     │ │
│               │                   │   │                │ │
│               └───────┬───────────┘   └───────┬────────┘ │
│                       │                       │          │
│                       ↓                       ↓          │
│                  routed_output          shared_output     │
│                       │                       │          │
│                       └───────── + ───────────┘          │
│                                 ↓                        │
│                          final_output                    │
│                                                          │
│  (vLLM 可用 CUDA multi-stream 并行执行两条路径)          │
└──────────────────────────────────────────────────────────┘
   ↓ (残差连接)
Output: hidden_states [num_tokens, 4096]
```

## 三、MoE Block 详细参数

### 代码中的关键配置
```python
# 从代码看到的结构
class Qwen2MoeSparseMoeBlock:
    self.gate = ReplicatedLinear(
        hidden_size,        # 4096
        num_experts,        # 64
    )
    
    self.shared_expert_gate = ReplicatedLinear(
        hidden_size,        # 4096
        1,                  # shared expert 的 gate
    )
    
    self.shared_expert = Qwen2MoeMLP(
        hidden_size=4096,
        intermediate_size=2048,  # shared expert 的中间层
    )
    
    self.experts = FusedMoEFactory(
        num_experts=64,              # 64 个 expert
        top_k=8,                     # 每个 token 选 8 个
        hidden_size=4096,
        intermediate_size=14336,     # 每个 expert 的中间层
    )
```

### 实际配置（从 config.json）
```json
{
  "hidden_size": 4096,
  "num_experts": 64,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 14336,
  "shared_expert_intermediate_size": 2048,
  "decoder_sparse_step": 1,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "num_hidden_layers": 28
}
```

## 四、关键数字解释

### 参数量计算

#### 每个 Expert 的参数
```
每个 Expert 是标准 FFN:
  gate_proj:  hidden → intermediate = 4096 × 14336 = 58M
  up_proj:    hidden → intermediate = 4096 × 14336 = 58M
  down_proj:  intermediate → hidden = 14336 × 4096 = 58M
  
  每个 Expert ≈ 174M 参数
```

#### MoE Layer 总参数
```
64 experts × 174M = 11.1B
+ Shared expert ≈ 50M
+ Router ≈ 0.3M
= 约 11.2B per MoE layer
```

#### 全模型参数
```
28 layers × (Attention + MoE)
Attention 每层 ≈ 100M
MoE 每层 ≈ 11.2B

总参数 ≈ 57B (hence the name)
```

### 激活参数计算

```
每个 token 激活：
  - Attention 层：100% 激活
  - MoE 层：8/64 = 12.5% 激活
  - Shared expert：100% 激活

实际计算量 ≈ 14B (hence "A14B")
```

## 五、与 Dense 模型对比

### Dense Model (如 Qwen2.5-7B)
```
每层结构：
  Attention + 单个 FFN
  
  FFN:
    gate_proj:  4096 → 11008
    up_proj:    4096 → 11008
    down_proj:  11008 → 4096
```

### MoE Model (Qwen2-57B-A14B)
```
每层结构：
  Attention + 64个 FFN (experts)
  
  每个 Expert:
    gate_proj:  4096 → 14336
    up_proj:    4096 → 14336
    down_proj:  14336 → 4096
```

### 对比表

| 维度 | Dense 7B | MoE 57B-A14B |
|-----|----------|--------------|
| 总参数 | 7B | 57B |
| 每层 FFN 数量 | 1 | 64 + 1 shared |
| 每次激活参数 | 7B (100%) | 14B (≈25%) |
| 内存占用 | ~14GB (FP16) | ~110GB (FP16) |
| 计算量/token | 固定 | 约 2× Dense 7B |

## 六、Week 6 学习会用到的部分

### Day 1-2: 理解架构
- ✅ 刚看的这个结构图
- Router 如何计算 expert 选择概率
- 为什么需要 64 个 expert

### Day 3-4: FusedMoE Kernel
- `self.experts = FusedMoEFactory(...)` 这行代码
- `fused_moe()` 函数如何执行 64 个 expert
- Token permutation 的作用

### Day 5: Expert Parallelism
- 如何将 64 个 expert 分配到多个 GPU
- All-to-All 通信如何传输 token

### Day 6-7: 部署调优
- 57B 参数需要多少显存
- 如何配置 TP/EP
- 性能优化参数

## 七、关键代码位置（供 Day 1 学习）

```python
# vllm/model_executor/models/qwen2_moe.py

# 行 121-189: Qwen2MoeSparseMoeBlock
#   - 行 137-143: Router (gate)
#   - 行 145-151: Shared expert gate
#   - 行 153-164: Shared expert
#   - 行 166-175: FusedMoE experts
#   - 行 177-189: forward() 流程

# 行 281-340: Qwen2MoeDecoderLayer
#   - 行 295-305: Attention
#   - 行 314-318: MoE layer (每隔 decoder_sparse_step=1 层)
#   - 行 320-326: Dense FFN (某些层用 dense 代替 MoE)
```

## 八、下载后如何查看完整配置

```bash
# 下载完成后，查看配置文件
cat /data/model/Qwen2-57B-A14B-Instruct/config.json | grep -E "num_experts|hidden_size|intermediate|top_k"

# 查看模型大小
du -sh /data/model/Qwen2-57B-A14B-Instruct/

# 查看权重文件
ls -lh /data/model/Qwen2-57B-A14B-Instruct/*.safetensors
```

---

**现在你理解了模型结构，可以开始下载了。下载期间可以先学习 Day 1 的理论部分。**
