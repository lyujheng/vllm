# Key-specific Delta Attention (KDA) 原理与 vLLM 实现

## 1. 概述

KDA (Key-specific Delta Attention) 是 Kimi-K3 模型使用的线性注意力变体，属于 Delta Rule 线性递推家族。它与 GDN (Gated Delta Net) 共享同一套 chunk 并行框架，**核心区别是 gate 从标量升级为向量**，使得隐状态矩阵的每一列可以独立衰减。

代表模型：Kimi-K3

vLLM 中存在两套 KDA 实现：
- **共享 FLA 层**：`vllm/third_party/flash_linear_attention/ops/kda.py`，直接扩展 FLA 的 GDN kernel
- **Kimi-K3 专用层**：`vllm/models/kimi_k3/nvidia/ops/third_party/kda/`，针对 Kimi-K3 做了深度 kernel 融合优化

---

## 2. 核心递推公式

### 2.1 GDN 递推（回顾）

$$
h_t = \exp(g_t) \cdot h_{t-1} + \beta_t \cdot (v_t - h_{t-1} \cdot k_t) \cdot k_t^\top
$$

其中 $g_t$ 是**标量**，`h` 的所有列以相同速率衰减。

### 2.2 KDA 递推

$$
h_t = \exp(\mathbf{g}_t) \cdot h_{t-1} + \beta_t \cdot (v_t - h_{t-1} \cdot k_t) \cdot k_t^\top
$$

$$
o_t = h_t \cdot q_t \cdot \text{scale}
$$

> **⚠️ 与 GDN 的关键区别**：$\mathbf{g}_t$ 从标量变为**向量** `[K]`，`exp(g_t)` 作用方式从整体缩放变为逐列缩放。

| 符号 | 形状 | 含义 | GDN 对应 |
|---|---|---|---|
| `h_t` | `[V, K]` | 隐状态矩阵 | 相同 |
| `q_t` | `[K]` | query 向量（L2 归一化后） | 相同 |
| `k_t` | `[K]` | key 向量（L2 归一化后） | 相同 |
| `v_t` | `[V]` | value 向量 | 相同 |
| **`g_t`** | **`[K]` 向量** | gate，控制隐状态各列独立衰减 | **GDN: 标量** |
| `β_t` | 标量 `(0,1)` | 写入强度 = `sigmoid(b)` | 相同 |
| `scale` | 标量 | `K^{-0.5}` | 相同 |

递推展开：

1. `exp(g_t) · h_{t-1}` — 隐状态**逐列**以不同速率衰减（GDN 是整体衰减）
2. `h_{t-1} · k_t` — 从旧状态中读出当前 key 方向的记忆
3. `v_t − (读出值)` — 计算 delta
4. `β_t · delta · k_t^⊤` — 把 delta 沿 key 方向写回
5. `h_t · q_t` — 用 query 从新状态中读出输出

### 2.3 Gate 的计算

> **⚠️ 与 GDN 的区别**：GDN 的 gate 是投影到 `[H]`（每 head 一个标量），KDA 的 gate 是投影到 `[H, K]`（每 head 每 key 维一个值）。

KDA 中 gate 的生成有两种模式：

**模式 1：softplus（标准模式）**

$$
\mathbf{g}_t = -\exp(A\_log) \cdot \text{softplus}(\mathbf{s}_t + \text{dt\_bias})
$$

其中 $\mathbf{s}_t \in \mathbb{R}^K$ 是 gate 投影的原始输出，`dt_bias` 是可学习的逐维偏置。

**模式 2：sigmoid bounded（lower_bound 模式，Kimi-K3 使用）**

$$
\mathbf{g}_t = \text{lower\_bound} \cdot \sigma(\exp(A\_log) \cdot (\mathbf{s}_t + \text{dt\_bias}))
$$

`lower_bound` 是负数（如 `-5.0 ≤ lower_bound < 0`），保证 gate 值有界，这对 chunk 算法的数值稳定性至关重要。

**代码**：`fused_kda_gate` / `fused_kda_gate_chunk_cumsum`

---

## 3. Gate 从标量变向量的连锁影响

这是理解 KDA 与 GDN 实现差异的核心。gate 维度变化导致以下算法层面的连锁改动：

### 3.1 隐状态衰减方式

| | GDN | KDA |
|---|---|---|
| gate 形状 | `[B, T, H]` — 标量 | `[B, T, H, K]` — 向量 |
| 衰减操作 | `h *= exp(g)` — 整个矩阵乘标量 | `h *= exp(g)[None, :]` — 逐列乘不同值 |

```python
# GDN decode
b_h *= exp(b_g)           # 标量乘

# KDA decode
b_h *= exp(b_gk[None, :])  # 向量按列乘
```

**代码**：`fused_recurrent.py` 中 `IS_KDA` 分支（共享 FLA 层，行 132-137）

### 3.2 Cumsum kernel

| | GDN | KDA |
|---|---|---|
| 输入 | `[B, T, H]` 标量 | `[B, T, H, K]` 向量 |
| kernel | `tl.cumsum(b_s, axis=0)` 一维累加 | 下三角矩阵 × 数据矩阵 `tl.dot(m_cumsum, b_gate)` |
| 输出 | `[B, T, H]` 每 chunk 内前缀和 | `[B, T, H, K]` 每 chunk 内每维前缀和 |

> **⚠️ 与 GDN 的区别**：GDN 用 `tl.cumsum`（一维原子操作），KDA 用下三角矩阵乘法实现向量 cumsum，因为需要对 K 维的每个分量独立累加。

```python
# GDN cumsum (cumsum.py - chunk_local_cumsum_scalar_kernel)
b_s = tl.load(p_s)
b_o = tl.cumsum(b_s, axis=0)

# KDA cumsum (kda.py - kda_gate_cumsum_fwd_kernel)
m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)  # 下三角矩阵
b_y = tl.dot(m_cumsum, b_gate) * cumsum_scale                  # 矩阵乘实现
```

### 3.3 依赖矩阵 A 的计算

> **⚠️ 与 GDN 的最大算法差异**：GDN 只需要 1 个 A 矩阵，KDA 需要 2 个。

| | GDN | KDA |
|---|---|---|
| 依赖矩阵 | 1 个 `A` = β·K·K^T·exp(衰减) | 2 个：`Akk`（用于三角求解）、`Aqk`（用于输出） |
| 原因 | `exp(cumg[i]-cumg[j])` 是标量，可以在 K·K^T 之后再乘 | `exp(gk[i]-gk[j])` 是向量，必须**先吸收进 Q/K 再做矩阵乘** |

**推导**：GDN 中 token i 和 j 之间的衰减 `exp(cumg[i] - cumg[j])` 是标量，可以独立于 K 维度计算：

$$
A_{GDN}[i,j] = \beta_i \cdot (k_i \cdot k_j) \cdot \exp(\text{cumg}[i] - \text{cumg}[j])
$$

但 KDA 中 `exp(cumgk[i] - cumgk[j])` 是 `[K]` 向量，必须先把衰减吸收进 Q 和 K：

$$
A_{kk}[i,j] = \beta_i \cdot \sum_d k_{i,d} \cdot \exp(\text{cumgk}_{i,d} - \text{cumgk}_{j,d}) \cdot k_{j,d}
$$

$$
A_{qk}[i,j] = \text{scale} \cdot \sum_d q_{i,d} \cdot \exp(\text{cumgk}_{i,d} - \text{cumgk}_{j,d}) \cdot k_{j,d}
$$

`Akk` 用于 delta rule 的三角求解（求等效 value），`Aqk` 用于最终输出计算。两者不能合并，因为 Q≠K。

**代码对比**：

```python
# GDN (chunk_scaled_dot_kkt.py) — 1 个 A 矩阵
b_kb = k * beta[:, None]
b_A = dot(b_kb, trans(k))  # K·K^T
b_A *= exp(cumg[:, None] - cumg[None, :])  # 标量衰减后乘

# KDA (kda.py / chunk_intra.py) — 2 个矩阵
# 先把向量衰减吸收进 K
b_k_gated = b_k * exp2(b_g - b_gn)     # K 吸收 gate
b_kt_gated = b_kt * exp2(b_gn - b_gk)  # K^T 吸收 gate
# 再做矩阵乘
b_Akk = dot(b_k_gated, b_kt_gated) * b_beta[:, None]  # 用于求解
b_Aqk = dot(b_q_gated, b_kt_gated) * scale             # 用于输出
```

---

## 4. 模型层调用链

以 Kimi-K3 为例（`vllm/models/kimi_k3/nvidia/kda.py`）：

```
KimiK3DeltaAttention.forward()
  │
  ├─ Part 1: Input Projection
  │    in_proj_qkvgfab → q, k, v, g_proj, f_a, beta (packed)
  │    f_b_proj(f_a)   → g1 (gate 投影, [H, K] 向量)  ← ⚠️ 与 GDN 不同
  │
  ├─ Part 2: Core Attention (_forward)
  │    ├─ causal_conv1d → 对 Q, K, V 分别做 1D 因果卷积
  │    │    (GDN 对 packed QKV 做单次卷积)
  │    │
  │    ├─ [prefill] chunk_kda_with_fused_gate()  ← chunk 并行算子
  │    │    或 _flashkda_prefill()               ← CUDA C++ 加速版
  │    ├─ [decode]  fused_recurrent_kda()        ← 逐 token 递推
  │    │    或 fused_recurrent_kda_packed_decode() ← packed decode
  │    │    或 ops.fused_kda_decode()            ← 全融合 CUDA kernel
  │    │
  │    └─ [spec decode] fused_recurrent_kda() with multi-token
  │
  └─ Part 3: Output
       FusedRMSNormGated(sigmoid gate) + o_proj
```

### 与 GDN 调用链的主要差异

| 方面 | GDN | KDA |
|---|---|---|
| Gate 投影 | 投影到 `[H]` 标量 | 投影到 `[H*K]` 向量，通过独立的 `f_b_proj` |
| Gate 参数 | `g`, 标量 per head | `g1` (raw_g, `[H, K]`), `A_log`, `dt_bias` |
| β 参数 | 独立投影 | 与 gate 打包投影，运行时取 sigmoid |
| 卷积 | 单次 packed QKV 卷积 | Q/K/V 分别卷积（prefill）或 packed（decode） |
| Prefill 算子 | `chunk_gated_delta_rule()` | `chunk_kda_with_fused_gate()` 或 `_flashkda_prefill()` |
| Decode 变体 | fused_recurrent / packed_decode | 3 种: recurrent / packed_decode / fused_kda_decode (CUDA) |
| Norm | RMSNormGated(swish) | FusedRMSNormGated(sigmoid) — activation 不同 |

---

## 5. Decode 实现

### 5.1 共享 FLA 层的 Decode（IS_KDA 分支）

`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py` 中通过 `IS_KDA` 编译常量区分 GDN 和 KDA：

```python
for i_t in range(T):
    b_q = load(q[t]) * scale
    b_k = load(k[t])
    b_v = load(v[t])

    if not IS_KDA:
        b_g = load(g[t])          # 标量
        b_h *= exp(b_g)           # 整体衰减
    else:
        b_gk = load(gk[t])       # [K] 向量
        b_h *= exp(b_gk[None, :]) # ⚠️ 逐列衰减

    b_v -= sum(b_h * b_k[None, :], 1)   # delta
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]  # 外积写回
    b_o = sum(b_h * b_q[None, :], 1)    # 读出
```

### 5.2 Kimi-K3 专用 Decode

Kimi-K3 有三种 decode kernel，按优先级选择：

1. **`ops.fused_kda_decode`**（CUDA C++）：把卷积、gate 计算、递推、RMSNorm 全部融合在一个 kernel 中。仅 SM90+/BF16/head_dim=128 支持。

2. **`fused_recurrent_kda_packed_decode`**：从 packed QKV 直接做递推，gate+beta 的 sigmoid/softplus 在 kernel 内计算（`USE_GATE_IN_KERNEL`）。

3. **`fused_recurrent_kda`**：通用 recurrent，支持 speculative decoding 的多 token 路径。

```python
# Kimi-K3 decode 中 gate+beta 在 kernel 内计算
if USE_GATE_IN_KERNEL:
    b_gate += b_bias                        # dt_bias
    b_a = exp(load(A_log))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * sigmoid(b_a * b_gate)   # bounded gate
    else:
        b_gate = -b_a * softplus(b_gate)                # softplus gate

b_state *= exp(b_gate[None, :])             # ⚠️ 向量逐列衰减
b_v -= sum(b_state * b_k[None, :], 1)
b_beta = sigmoid(b_beta)                    # APPLY_BETA_SIGMOID
b_v *= b_beta
b_state += b_v[:, None] * b_k[None, :]
```

---

## 6. Prefill 实现：Chunk 并行算法

KDA 的 chunk 算法流程与 GDN 相同（6 步流水线），但多个步骤因 gate 向量化而改动。

### 6.1 整体流程对比

```
GDN:                                    KDA:
Step 1: scalar cumsum                   Step 1: vector cumsum (fused gate+cumsum)  ⚠️
Step 2: A = β·K·K^T·exp(衰减)           Step 2: Akk, Aqk 分别计算               ⚠️
Step 3: solve_tril(A)                   Step 3: solve_tril(Akk)                   相同
Step 4: w, u = A^{-1} · (β·k, β·v)     Step 4: w, u = Akk^{-1} · (β·k, β·v)    相同
Step 5: chunk 间递推 H                   Step 5: chunk 间递推 H                    相同
Step 6: o = q·H + Aqk·v_new            Step 6: o = q·exp(g)·H + Aqk·v_new       ⚠️
```

### 6.2 Step 1: Gate 计算 + 向量 Cumsum

> **⚠️ 与 GDN 的区别**：KDA 融合了 gate 激活和向量 cumsum，同时输出 `beta = sigmoid(raw_beta)`。

```python
# Kimi-K3 专用层 (chunk.py - kda_gate_chunk_cumsum_vector_kernel)
# 每个 (chunk, head) 启动 cdiv(K, BS)+1 个 program:
#   program 0: 计算 beta = sigmoid(raw_beta)
#   program 1..N: 计算 gate + cumsum

# gate 激活（与 decode 相同的公式）
if USE_LOWER_BOUND:
    b_gate = lower_bound * sigmoid(b_a * b_s)
else:
    b_gate = -b_a * softplus(b_s + b_bias)

# 向量 cumsum — 直接逐元素累加（因为 gate 是 [K] 向量）
b_o = tl.cumsum(b_gate, axis=0) * cumsum_scale  # 沿 token 轴累加

# 共享 FLA 层 (kda.py - kda_gate_cumsum_fwd_kernel)
# 用下三角矩阵乘法实现，等效但更适合较大 K
m_cumsum = where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
b_y = dot(m_cumsum, b_gate) * cumsum_scale
```

**为什么乘 `cumsum_scale = RCP_LN2`**：后续 kernel 用 `exp2` 而非 `exp` 计算衰减，需要提前做 `ln→log2` 转换：`exp(x) = exp2(x / ln(2))`。

### 6.3 Step 2: 构建依赖矩阵（Akk + Aqk）

> **⚠️ 与 GDN 的核心区别**：需要同时计算两个矩阵。

GDN 中 `exp(cumg[i] - cumg[j])` 是标量，可在 `K·K^T` 之后乘。KDA 中它是向量，必须先吸收进 Q/K：

```python
# 对每对子 chunk (i, j), i > j:
b_gn = load(g[子chunk_i 起始位置])        # [BK] 参考点

# 把向量衰减吸收进 K
b_k_gated = b_k * exp2(b_g - b_gn)       # [BC, BK], row i 的 K
b_kt_gated = b_kt * exp2(b_gn - b_gk)    # [BK, BC], col j 的 K^T

# 并行计算两个矩阵
b_Akk += dot(b_k_gated, b_kt_gated)      # [BC, BC], 用于三角求解
b_Aqk += dot(b_q_gated, b_kt_gated)      # [BC, BC], 用于输出

# Akk 事后乘 beta
b_Akk *= b_beta[:, None]
```

**Kimi-K3 的融合优化**：`chunk_kda_fwd_kernel_inter_solve_fused` 把 Step 2 的 inter 部分（子 chunk 间系数）和 Step 3 的三角求解融合到一个 kernel 中，减少 kernel launch 和全局内存读写。

### 6.4 Step 3: 三角求解 (I+Akk)⁻¹

与 GDN 完全相同的分层策略：64×64 矩阵分成 4×4 个 16×16 子块，先独立求解对角块的逆，再用分块矩阵逆公式合并。

> **⚠️ Kimi-K3 的区别**：把 Step 2 inter 和 Step 3 融合在 `chunk_kda_fwd_kernel_inter_solve_fused` 中。intra（对角块）在 `chunk_kda_fwd_kernel_intra_sub_chunk` 中单独计算，输出到 `Akkd` (fp32)，然后 fused kernel 从 `Akkd` 加载对角块、计算非对角块、做三角求解、最终写出完整的 `Akk_inv`。

### 6.5 Step 4: 计算等效 value (u) 和等效 key (w)

与 GDN 相同的公式和实现：

$$
u = (I+Akk)^{-1} \cdot (\beta \cdot v)
$$

$$
w = (I+Akk)^{-1} \cdot (\beta \cdot \exp(\text{cumgk}) \cdot k)
$$

> **⚠️ 与 GDN 的细微区别**：GDN 中 `exp(g)` 是标量乘进去，KDA 中 `exp(cumgk)` 是向量逐元素乘 k。但因为 `(I+A)^{-1}` 作用在 token 维度上，K 维度不变，所以 kernel 结构一致。

同时还计算 `kg = k * exp(gn - cumgk)`（gate 衰减后的 key），供 Step 5 使用。

### 6.6 Step 5: chunk 间状态递推

与 GDN 完全相同。遍历每个 chunk，串行更新隐状态 H：

```python
H = initial_state
for i_t in range(NT):
    store(h_out[i_t], H)
    correction = w[i_t] @ H^T
    v_new = u[i_t] - correction
    v_new *= exp(g_last - cumg)
    H *= exp(g_last)
    H += (k[i_t]^T @ v_new)^T
```

> 注意：Step 5 对 `gk` 的使用是 `exp2(gk)` 而非 `exp(gk)`，因为 Step 1 已乘过 `RCP_LN2`。

### 6.7 Step 6: 计算输出

> **⚠️ 与 GDN 的区别**：inter-chunk 部分需要把向量 gate 吸收进 query。

$$
o_t = \underbrace{\text{scale} \cdot (q_t \cdot \exp(\text{cumgk}_t)) \cdot H[\text{chunk}]^\top}_{\text{inter-chunk}} + \underbrace{A_{qk}[t, :] \cdot v\_new}_{\text{intra-chunk}}
$$

GDN 中 inter-chunk 是 `q * exp(cumg) * scale`（标量乘），KDA 中是 `q * exp(cumgk) * scale`（向量逐元素乘）：

```python
# inter-chunk
b_qg = (b_q * exp2(b_g)).to(b_q.dtype)    # ⚠️ b_g 是 [BT, BK] 向量
b_o = dot(b_qg, trans(b_h))               # [BT, BV]

# intra-chunk: 直接用预计算的 Aqk
b_A = load(Aqk[chunk])                     # [BT, BT]，已在 Step 2 计算好
b_A = where(causal_mask, b_A, 0)
b_o += dot(b_A, b_v)                       # [BT, BV]
```

GDN 的 Step 6 需要在 kernel 内重新计算 intra 的 `Q·K^T · exp(衰减)` 矩阵，而 KDA 直接加载 Step 2 预计算的 `Aqk`，减少了 Step 6 的计算量。

---

## 7. Prefill vs Decode 对比

| | Decode | Prefill (Chunk) |
|---|---|---|
| 序列长度 | T=1（或少量 token）| T 可达数千 |
| 算法 | 逐 token 递推 | 分块并行 + chunk 间递推 |
| 串行步数 | T | T/C（C=64）|
| gate 处理 | 直接逐步 exp(g) | cumsum + exp2 |
| Kernel 数量 | 1-3 个融合 kernel | 5-6 个 kernel 流水线 |

---

## 8. GDN vs KDA 实现差异总结

| 维度 | GDN | KDA |
|---|---|---|
| **Gate 形状** | `[B, T, H]` — 标量 | `[B, T, H, K]` — 向量 |
| **Gate 计算** | `-exp(A_log) * softplus(a + dt_bias)` | 同公式，但输出是 `[K]` 向量；Kimi-K3 还支持 sigmoid bounded |
| **递推衰减** | `h *= exp(g)` 整体缩放 | `h *= exp(gk)[None, :]` 逐列缩放 |
| **Cumsum** | 标量 `tl.cumsum` | 向量 `tl.cumsum`（逐分量）或下三角矩阵乘 |
| **依赖矩阵** | 1 个 A | 2 个 Akk + Aqk |
| **Step 2 原因** | 标量衰减可后乘 | 向量衰减必须先吸收进 Q/K 再做矩阵乘 |
| **三角求解** | `solve_tril(A)` | `solve_tril(Akk)` — 对 Akk 求解，Aqk 不需要求解 |
| **Step 6 输出** | kernel 内重新计算 Q·K^T | 直接加载 Step 2 预计算的 Aqk |
| **Kernel 融合** | 各步独立 kernel | Kimi-K3：Step 2 inter + Step 3 融合为一个 kernel |
| **Decode 变体** | fused_recurrent / packed_decode | 额外增加全融合 CUDA C++ decode kernel |
| **输出归一化** | RMSNormGated(swish) | FusedRMSNormGated(sigmoid) |
| **Q/K norm** | 可选 L2 norm | 默认启用 L2 norm |
| **β 处理** | 独立 sigmoid | Kimi-K3 在 fused gate kernel 中同时计算 |
| **代表模型** | Qwen3-Next, Qwen3.5 | Kimi-K3 |

---

## 9. 性能关键设计

1. **两套实现共存**：共享 FLA 层提供通用 KDA 支持，Kimi-K3 专用层做深度优化
2. **Step 2+3 kernel 融合**：Kimi-K3 把 inter Akk 计算和三角求解融合，减少全局内存读写
3. **Step 2 预计算 Aqk**：KDA 在 Step 2 就计算好 Aqk，Step 6 直接加载，避免重复计算
4. **全融合 decode kernel (CUDA C++)**：把 conv1d + gate + recurrent + norm 全部融合，零中间张量分配
5. **Gate 融合进 decode kernel**：`USE_GATE_IN_KERNEL` 避免单独 materialize gate tensor
6. **PDL (Programmatic Dependent Launch)**：SM90+ 上支持 kernel 间依赖启动，减少 launch overhead
7. **FlashKDA 后端**：SM90+ 上的 CUDA C++ prefill 实现，作为 Triton 的替代
8. **intra sub-chunk token-parallel**：对角块的 Akk 计算支持 token 级并行

---

## 10. 代码文件索引

### 共享 FLA 层 (`vllm/third_party/flash_linear_attention/ops/`)

| 文件 | 功能 | KDA 变化 |
|---|---|---|
| `kda.py` | KDA 的 chunk 流水线入口 + gate 融合 + cumsum | 新增文件 |
| `cumsum.py` | chunk 内 gate 前缀和（GDN scalar + KDA vector）| 新增 vector kernel |
| `chunk_scaled_dot_kkt.py` | Step 2: 构建依赖矩阵 | GDN 专用，KDA 在 kda.py 中重写 |
| `solve_tril.py` | Step 3: 三角求解 | GDN/KDA 共用 |
| `wy_fast.py` | Step 4: WY 表示 | GDN 专用，KDA 在 kda.py 中重写 |
| `chunk_delta_h.py` | Step 5: chunk 间递推 | GDN/KDA 共用 |
| `chunk_o.py` | Step 6: 计算输出 | GDN 专用，KDA 在 kda.py 中重写 |
| `fused_recurrent.py` | Decode 递推（IS_KDA 分支）| 共用 kernel，编译分支 |

### Kimi-K3 专用层 (`vllm/models/kimi_k3/nvidia/ops/third_party/kda/`)

| 文件 | 功能 |
|---|---|
| `chunk.py` | prefill 流水线入口 + fused gate cumsum + recompute_w_u + output |
| `chunk_intra.py` | Step 2+3 融合：intra sub-chunk + inter + solve_tril |
| `chunk_intra_token_parallel.py` | 对角块 Akk 的 token-parallel 计算 |
| `fused_recurrent.py` | decode 递推：recurrent / packed_decode / gate+beta fused |

### 模型层

| 文件 | 功能 |
|---|---|
| `vllm/models/kimi_k3/nvidia/kda.py` | `KimiK3DeltaAttention` 模型层，调度 prefill/decode |
| `vllm/models/kimi_k3/nvidia/kda_metadata.py` | metadata builder，继承 GDN 但跳过 FLA chunk 索引构建 |
