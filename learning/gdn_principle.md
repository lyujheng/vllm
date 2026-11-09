# Gated Delta Net (GDN) 原理与 vLLM 实现

## 1. 概述

Gated Delta Net (GDN) 是一种**线性注意力 + 递推状态**的序列建模架构，属于 SSM (State Space Model) / 线性 RNN 家族，与 Mamba 同类。它不使用 softmax attention，而是通过维护一个递推隐状态矩阵来记忆历史信息。

代表模型：Qwen3-Next、Qwen3.5、OLMo 等。

vLLM 中的实现来自开源项目 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)（作者 Songlin Yang & Yu Zhang），代码位于 `vllm/third_party/flash_linear_attention/` 目录。

### GDN 与其他架构的区别

| 架构 | 代表模型 | 注意力类型 | 算子 |
|---|---|---|---|
| MLA | DeepSeek-V2/V3 | softmax attention（低秩 KV）| FlashMLA, FlashInfer |
| GDN | Qwen3-Next | 线性注意力 + delta rule 递推 | FLA (flash-linear-attention) |
| KDA | Kimi-K3 | 线性注意力 + key-specific delta rule | FLA 的 KDA 变体 |
| Mamba | Mamba-1/2 | SSM 递推 | Mamba kernels |

---

## 2. 核心递推公式

GDN 的核心是一个逐 token 的递推，维护一个 `[V, K]` 的隐状态矩阵 `h`：

$$
h_t = \exp(g_t) \cdot h_{t-1} + \beta_t \cdot (v_t - h_{t-1} \cdot k_t) \cdot k_t^\top
$$

$$
o_t = h_t \cdot q_t \cdot \text{scale}
$$

各符号含义：

| 符号 | 形状 | 含义 | 来源 |
|---|---|---|---|
| `h_t` | `[V, K]` | 隐状态矩阵，记忆历史信息 | 递推计算 |
| `q_t` | `[K]` | query 向量 | 线性投影 + L2 norm |
| `k_t` | `[K]` | key 向量 | 线性投影 + L2 norm |
| `v_t` | `[V]` | value 向量 | 线性投影 |
| `g_t` | 标量 | gate（log 空间），控制旧状态的遗忘速度 | `-exp(A_log) * softplus(a + dt_bias)` |
| `β_t` | 标量 `(0,1)` | 写入强度，控制新信息的写入力度 | `sigmoid(b)` |
| `scale` | 标量 | 缩放因子 | `K^{-0.5}` |

递推展开理解：

1. `exp(g_t) · h_{t-1}` — 用 gate 衰减旧状态
2. `h_{t-1} · k_t` — 从旧状态中"读出"当前 key 方向的记忆
3. `v_t − (读出值)` — 计算 **delta**（当前 value 与旧记忆的差）
4. `β_t · delta · k_t^⊤` — 把 delta 沿 key 方向写回
5. `h_t · q_t` — 用 query 从新状态中读出输出

这就是 "Delta Rule" 的含义：不是直接覆盖写入，而是只写入差量，类似 Hopfield 网络的误差修正学习规则。

---

## 3. 模型层调用链

以 Qwen3-Next 为例（`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`）：

```
QwenGatedDeltaNetAttention.forward_cuda()
  │
  ├─ Part 1: Input Projection
  │    in_proj_qkvz → mixed_qkv (packed q,k,v)
  │    in_proj_ba → b, a (gate 参数)
  │
  ├─ Part 2: Core Attention
  │    ├─ causal_conv1d → 1D 因果卷积
  │    ├─ fused_post_conv_prep → 拆分 q,k,v + L2 norm + 计算 g,β
  │    │    g = -exp(A_log) * softplus(a + dt_bias)
  │    │    β = sigmoid(b)
  │    │
  │    ├─ [prefill] chunk_gated_delta_rule()    ← chunk 并行算子
  │    └─ [decode]  fused_recurrent / packed_decode  ← 逐 token 递推算子
  │
  └─ Part 3: Output Projection
       RMSNormGated + output linear
```

---

## 4. Decode 实现：逐 token 递推

Decode 阶段每次只处理 1 个 token（或少量 token），直接执行递推公式。

代码位置：`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`

核心 kernel 代码（简化）：

```python
for i_t in range(T):
    b_q = load(q[t])
    b_k = load(k[t])
    b_v = load(v[t])
    b_q = b_q * scale

    b_h *= exp(b_g)                        # gate 衰减旧状态
    b_v -= sum(b_h * b_k[None, :], 1)      # delta = v - h·k
    b_v *= b_beta                           # delta *= β
    b_h += b_v[:, None] * b_k[None, :]     # h += delta · k^T（外积）
    b_o = sum(b_h * b_q[None, :], 1)       # o = h · q
    store(o[t], b_o)
```

decode 时 T=1，一步到位，整个递推融合在单个 Triton kernel 中。

---

## 5. Prefill 实现：Chunk 并行算法

### 5.1 为什么需要 chunk 算法

Prefill 阶段 T 可达数千，逐 token 串行太慢。Chunk 算法的核心思想是：

- 把长度 T 的序列切成大小 C=64 的 chunk
- **chunk 内**：通过代数变换把串行递推转化为并行矩阵运算
- **chunk 间**：串行递推，但只有 T/C 步

代码位置：`vllm/third_party/flash_linear_attention/ops/chunk.py`

```python
def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, ...):
    # Step 1: 预处理 gate
    g = chunk_local_cumsum(g, chunk_size=64)

    # Step 2: 构建 chunk 内依赖矩阵
    A = chunk_scaled_dot_kkt_fwd(k, beta, g)

    # Step 3: 三角求解
    A = solve_tril(A)

    # Step 4: 计算等效 value 和等效 key
    w, u = recompute_w_u_fwd(k, v, beta, A, g)

    # Step 5: chunk 间状态递推
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)

    # Step 6: 计算输出
    o = chunk_fwd_o(q, k, v_new, h, g, scale)
```

### 5.2 Step 1: chunk 内 gate 前缀和

**目的**：把 gate 的连乘转化为减法，使得后续可以 O(1) 算出任意两个 token 之间的衰减系数。

**原理**：原始递推中，token s 到 token t 的累积衰减是连乘：

$$
\prod_{i=s+1}^{t} \exp(g_i) = \exp\left(\sum_{i=s+1}^{t} g_i\right)
$$

定义 chunk 内前缀和 `cumg[t] = g_1 + g_2 + ... + g_t`，则：

$$
\text{衰减}(s \to t) = \exp(\text{cumg}[t] - \text{cumg}[s])
$$

一次减法 + 一次 exp 即可，完全不需要串行累积。

**关键点**：只做 chunk 内的局部 cumsum，不跨 chunk。跨 chunk 的衰减由 Step 5 处理。

**示例**（chunk 内 4 个 token）：

```
原始 gate:   g = [-0.5, -0.3, -0.7, -0.2]
chunk cumsum: cumg = [-0.5, -0.8, -1.5, -1.7]

token 1 → token 3 的衰减:
  exp(cumg[3] - cumg[1]) = exp(-1.7 - (-0.8)) = exp(-0.9) ≈ 0.4066
```

**代码**：`vllm/third_party/flash_linear_attention/ops/cumsum.py`

```python
# scalar kernel（GDN，gate 形状 [B,T,H]）
b_s = tl.load(p_s)          # 加载 64 个 gate 值
b_o = tl.cumsum(b_s, axis=0) # 一维前缀和
```

全局衰减通过两层分工恢复：

```
token s → token t 的全局衰减 = (chunk 间衰减) × (chunk 内衰减)
  chunk 间: Step 5 逐 chunk 乘 exp(chunk 末尾 cumsum)
  chunk 内: Step 2/6 用 cumsum 做差
```

### 5.3 Step 2: 构建 chunk 内依赖矩阵 A

**目的**：计算 chunk 内 token 之间通过 delta rule 递推产生的相互影响系数矩阵。

**推导**：把递推公式改写为矩阵形式：

$$
h_t = [\exp(g_t) \cdot I - \beta_t \cdot k_t \cdot k_t^\top] \cdot h_{t-1} + \beta_t \cdot v_t \cdot k_t^\top
$$

展开 chunk 内所有 token 的递推后，定义"等效 value" `ṽ`（每个 token 经过 delta rule 修正后实际写入隐状态的值），它们满足下三角方程组：

```
ṽ₀ =                                      β₀·v₀
ṽ₁ = −A[1,0]·ṽ₀                         + β₁·v₁
ṽ₂ = −A[2,0]·ṽ₀ − A[2,1]·ṽ₁           + β₂·v₂
```

即 `(I + A) · ṽ = β·v`，其中：

$$
A[i,j] = \beta_i \cdot (k_i \cdot k_j) \cdot \exp(\text{cumg}[i] - \text{cumg}[j]) \quad (i > j)
$$

**为什么 A 的元素是标量**：`h · k` 是矩阵乘向量，其中 `h` 是外积叠加 `Σ ṽ_j · k_j^⊤`，所以 `h · k_i = Σ (k_j · k_i) · ṽ_j`。内积 `k_j · k_i` 把 K 维消掉了，结果是标量。

**计算方式**：一次矩阵乘并行算出所有 C×C 个系数。

```python
b_kb = k * beta[:, None]                  # [C, K] 每行乘 β
b_A = dot(b_kb, trans(k))                 # [C, K] @ [K, C] = [C, C]
b_A = b_A * exp(cumg[:, None] - cumg[None, :])  # 乘衰减
b_A = where(i > j, b_A, 0)               # 下三角 mask
```

**代码**：`vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py`

### 5.4 Step 3: 三角求解 (I+A)⁻¹

**目的**：求解 `(I + A)⁻¹`，为 Step 4 做准备。

`I + A` 是单位下三角矩阵（对角线全为 1 的下三角矩阵）：

```
        ┌ 1                        ┐
I + A = │ A[1,0]    1              │
        │ A[2,0]    A[2,1]    1    │
        └ A[3,0]    A[3,1]    A[3,2]  1 ┘
```

**求解方法**：分层策略。

对 64×64 的矩阵，不能直接在 GPU 寄存器中处理（需要 16KB 寄存器）。分成 4×4 个 16×16 的子块：

**阶段一**：独立求解 4 个对角块的逆（前代法，每个 16×16 只需 14 步迭代）

```
Ai₁₁ = (I + A₁₁)⁻¹,  Ai₂₂ = (I + A₂₂)⁻¹,  Ai₃₃ = (I + A₃₃)⁻¹,  Ai₄₄ = (I + A₄₄)⁻¹
```

前代法核心（对 16×16 块）：

```python
b_A = -where(m_lower_tri, b_A, 0)    # 取下三角并取负
for i in range(2, 16):
    b_a = -load(A 的第 i 行)
    b_a = b_a + sum(b_a[:, None] * b_A, 0)   # 利用已求出的行修正当前行
    b_A = where(row == i, b_a, b_A)           # 写回
b_A += I                                       # 加单位矩阵
```

**阶段二**：用分块矩阵逆公式合并非对角块

由 `M · M⁻¹ = I` 的分块展开，对 i > j：

$$
Ai_{ij} = -Ai_{ii} \cdot \sum_{k=j}^{i-1} A_{ik} \cdot Ai_{kj}
$$

按对角线距离从近到远计算：

```
距离 1: Ai₂₁ = −Ai₂₂ · A₂₁ · Ai₁₁
距离 2: Ai₃₁ = −Ai₃₃ · (A₃₁·Ai₁₁ + A₃₂·Ai₂₁)
距离 3: Ai₄₁ = −Ai₄₄ · (A₄₁·Ai₁₁ + A₄₂·Ai₂₁ + A₄₃·Ai₃₁)
```

每一步都是 16×16 的 `tl.dot`，利用 tensor core 高效执行。

**代码**：`vllm/third_party/flash_linear_attention/ops/solve_tril.py`

### 5.5 Step 4: 计算等效 value (u) 和等效 key (w)

**目的**：用 `(I+A)⁻¹` 解方程，得到 chunk 内每个 token 的等效量，消解 chunk 内 token 间的递推依赖。

$$
u = (I+A)^{-1} \cdot (\beta \cdot v) \quad \text{（等效 value，代码中叫 u）}
$$

$$
w = (I+A)^{-1} \cdot (\beta \cdot \exp(g) \cdot k) \quad \text{（等效 key，代码中叫 w）}
$$

`u` 和 `w` 的形状分别与 `v` `[B,T,H,V]` 和 `k` `[B,T,H,K]` 相同。变换矩阵 `(I+A)⁻¹` 作用在 chunk 内 token 维度上，H、V、K 维度不变。

```python
b_vb = b_v * b_beta[:, None]              # β·v, [C, V]
b_u = tl.dot(b_A, b_vb)                   # (I+A)⁻¹ · (β·v), [C, V]

b_kb = b_k * b_beta[:, None] * b_g[:, None]  # β·exp(g)·k, [C, K]
b_w = tl.dot(b_A, b_kb)                      # (I+A)⁻¹ · (β·exp(g)·k), [C, K]
```

`u` 的物理含义：每个 token 经过 delta rule 修正后实际写入隐状态的"等效 value"。
`w` 的物理含义：对应的"等效 key"，额外吸收了 β 和 gate 衰减。

有了 u 和 w，chunk 内 token 之间不再有递推依赖，后续步骤可以把 chunk 当作独立单元处理。

**代码**：`vllm/third_party/flash_linear_attention/ops/wy_fast.py`

### 5.6 Step 5: chunk 间状态递推

**目的**：计算每个 chunk 开始前的隐状态矩阵 `H[i]`，形状 `[V, K]`。

这是整个 chunk 算法中**唯一必须串行**的部分，但只有 T/C 步（而非 T 步）。

```python
H = initial_state  # [V, K]，可为零

for i_t in range(NT):            # 遍历每个 chunk
    store(h_out[i_t], H)          # ① 存储当前 h，给 Step 6 用

    correction = w[i_t] @ H^T     # ② [C,K] @ [K,V] = [C,V]
    v_new = u[i_t] - correction   # ③ delta 修正：减去旧记忆

    v_new *= exp(g_last - cumg)   # ④ chunk 内各 token 相对衰减
    H *= exp(g_last)              # ⑤ 旧状态整体衰减

    H += (k[i_t]^T @ v_new)^T    # ⑥ 累加当前 chunk 的贡献
```

**示例**（2 个 chunk）：

```
chunk 0:
  H[0] = 0（初始状态）
  correction = w · 0^T = 0（无历史）
  v_new = u - 0 = u
  H 更新为 chunk 0 的累积结果

chunk 1:
  H[1] = H（chunk 0 的结果）
  correction = w · H^T（从旧记忆中读出信息）
  v_new = u - correction（扣除旧记忆中已有的部分）
  H 继续更新
```

**输出**：
- `h_out[0..NT-1]`：每个 chunk 开始前的隐状态，给 Step 6 用
- `v_new`：delta 修正后的值（也给 Step 6 用）

**代码**：`vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py`

### 5.7 Step 6: 计算输出

**目的**：利用 Step 5 的隐状态和 chunk 内的 intra-attention，计算每个 token 的最终输出。

每个 token 的输出分两部分：

$$
o_t = \underbrace{\text{scale} \cdot q_t \cdot \exp(\text{cumg}_t) \cdot H[\text{chunk}]^\top}_{\text{inter-chunk: 历史所有 chunk 的贡献}} + \underbrace{\text{scale} \cdot \sum_{j \leq t} \exp(\text{cumg}_t - \text{cumg}_j) \cdot (q_t \cdot k_j) \cdot v\_new_j}_{\text{intra-chunk: 当前 chunk 内的贡献}}
$$

```python
# inter-chunk: q 与 chunk 初始隐状态的乘积
b_o = dot(b_q, trans(b_h))                         # [C,K] @ [K,V] = [C,V]

# intra-chunk: chunk 内的 Q·K 注意力
b_A = dot(b_q, b_k)                                # [C,K] @ [K,C] = [C,C]

# 乘以 gate 衰减
b_o = b_o * exp(cumg)[:, None]                      # inter: token 相对 chunk 起点的衰减
b_A = b_A * exp(cumg[:, None] - cumg[None, :])      # intra: token 对之间的相对衰减
b_A = where(i >= j, b_A, 0)                         # 因果 mask

# 合并
o = b_o * scale + dot(b_A, v_new) * scale
```

**代码**：`vllm/third_party/flash_linear_attention/ops/chunk_o.py`

---

## 6. Prefill vs Decode 对比

| | Decode | Prefill (Chunk) |
|---|---|---|
| 序列长度 | T=1（或很短）| T 可达数千 |
| 算法 | 逐 token 递推 | 分块并行 + chunk 间递推 |
| 串行步数 | T | T/C（C=64）|
| 计算复杂度 | O(T·V·K) | O(T·C·K + T/C·V·K) |
| Kernel 数量 | 1 个融合 kernel | 6 个独立 kernel 流水线 |
| 数学变换 | 无，直接递推 | 下三角求解 + WY 表示 |

---

## 7. GDN vs KDA

KDA (Key-specific Delta Attention) 是 Kimi-K3 使用的变体，与 GDN 的核心区别是 **gate 的粒度**：

| | GDN | KDA |
|---|---|---|
| gate 形状 | `[B, T, H]` — 每 head 一个标量 | `[B, T, H, K]` — 每 head 每 key 维一个值 |
| 递推 | `h *= exp(g)`（整体衰减）| `h *= exp(gk)`（逐列衰减）|
| cumsum kernel | scalar: `tl.cumsum` | vector: 下三角矩阵 × 数据矩阵 |
| 依赖矩阵 | 1 个 A 矩阵 | 2 个（Aqk 用于输出，Akk 用于三角求解）|
| 代表模型 | Qwen3-Next | Kimi-K3 |

gate 从标量变为向量后，`exp(gk[i] - gk[j])` 不再是标量，不能在内积之后才乘。必须先把 gate 吸收进 Q/K（`q * exp(gk)`, `k * exp(-gk)`），再做矩阵乘，因此需要分别计算 Q-K 和 K-K 的注意力矩阵。

---

## 8. 性能关键设计

1. **全部 Triton 实现**：跨 NVIDIA/AMD 平台
2. **Chunk 算子用 autotune**：自动搜索 BK、BV、num_warps、num_stages 的最优组合
3. **Decode 算子深度融合**：packed decode kernel 把 QKV 拆分、gate 计算、状态更新、输出计算融合在一个 kernel
4. **支持 continuous batching**：通过 `ssm_state_indices` 间接索引隐状态，兼容 vLLM 的 PagedAttention 式状态管理
5. **分层三角求解**：64×64 → 4 个 16×16 独立求解 + 分块矩阵逆合并，适配 GPU 寄存器限制
6. **chunk 内/间分治**：chunk 内前缀和 + chunk 间递推，全局衰减通过两层乘积恢复

---

## 9. 代码文件索引

| 文件 | 功能 |
|---|---|
| `ops/cumsum.py` | Step 1: chunk 内 gate 前缀和 |
| `ops/chunk_scaled_dot_kkt.py` | Step 2: 构建依赖矩阵 A = β·K·K^T·exp(衰减) |
| `ops/solve_tril.py` | Step 3: 分层三角求解 (I+A)⁻¹ |
| `ops/wy_fast.py` | Step 4: WY 表示，计算等效 value u 和等效 key w |
| `ops/chunk_delta_h.py` | Step 5: chunk 间状态递推 |
| `ops/chunk_o.py` | Step 6: 计算最终输出 |
| `ops/fused_recurrent.py` | Decode: 逐 token 融合递推 |
| `ops/chunk.py` | Prefill: 6 步流水线的编排入口 |
| `ops/utils.py` | 工具函数、平台检测、常量定义 |
| `ops/index.py` | 变长 batch 的 chunk 索引构建 |

所有文件路径前缀：`vllm/third_party/flash_linear_attention/`
