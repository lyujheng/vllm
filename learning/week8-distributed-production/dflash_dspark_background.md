# Day 5: DFlash + DSpark — 生产级投机解码

## 1. 回顾：传统投机解码的瓶颈

Week 5 学过三种投机解码方法，各有明确的局限：

**Eagle（自回归 draft）**：
- draft 模型逐 token 生成：`token_1 → token_2 → token_3 → ...`
- 每个 draft token 需要一次独立的 forward pass
- draft 质量高（因为有完整的自回归依赖），但 **draft 过程本身就是 memory-bound 的串行操作**
- N 个 draft token 需要 N 次 forward，draft 开销随 N 线性增长

**Medusa（并行多 head）**：
- 多个独立的 head 并行预测不同位置的 token
- 一次 forward 出 N 个 draft token，速度极快
- 但各位置之间**完全没有依赖**：position 3 的预测不知道 position 2 实际采了什么
- 结果：接受率随 draft 长度快速衰减，实际可用的 draft 长度有限

**N-gram**：
- 直接从 prompt 中匹配已有的 n-gram 模式
- 无需额外模型，零额外计算
- 但只能匹配已有模式，对新内容束手无策

核心矛盾：**速度和质量的 trade-off**。自回归保证 token 间依赖但慢，并行快但丢失依赖。

## 2. DFlash：并行 Draft 的基础架构

DFlash 是 DeepSeek 提出的并行 draft 方案，后来被 vLLM 原生实现。其核心思路是：**一次 forward 生成整个 draft block，不做逐 token 自回归**。

### 2.1 核心概念

**Anchor Token（锚点 token）**：
- 上一轮验证通过的最后一个 token，作为本轮 draft 的起点
- DFlash 中 anchor 是 "bonus token"（额外附加的 token），不参与采样
- 布局：`[anchor, mask_1, mask_2, ..., mask_N]`，共 1+N 个 query token
- anchor 提供已知的上下文信息，mask token 是需要预测的位置

**Parallel Drafting Token（掩码 token）**：
- 预测位置填入一个特殊的 mask token id（类似 BERT 的 `[MASK]`）
- 所有 mask token **同时**参与一次 forward pass
- 模型在每个 mask 位置输出一个 hidden state，用于预测该位置的 token

**Context-KV Precompute（上下文 KV 预计算）**：
- draft 模型不重新处理整个上下文序列
- 而是直接从目标模型（target model）的 hidden states 投影出 KV Cache
- 这一步在 CUDA graph 之外单独执行（因为 context 长度每步都变）

**非因果 Attention（Non-causal）**：
- 关键区别：draft block 内部的 mask token 之间**不做因果遮掩**
- 即 mask_3 可以 attend 到 mask_1 和 mask_2（虽然它们都是未知的 mask）
- 但 mask tokens 对 context KV 仍然是因果的（只能看到之前的 context）
- 这是 DFlash 命名中 "Flash" 的含义之一：一次 flash forward 出所有位置

### 2.2 DFlash 一轮 Draft 的完整流程

```
Step 1: Context-KV Precompute（不在 CUDA graph 中）
  - 输入：target model 的 hidden_states（本轮所有 context token）
  - 操作：draft 模型的 KV 投影层对 context 做 GEMM + RoPE
  - 输出：context KV 直接写入 draft 的 KV Cache

Step 2: Query-Block Forward（在 CUDA graph 中）
  - 输入：[anchor_token, mask, mask, ..., mask]，共 1+N 个 token
  - Attention：query tokens 对 context KV 做 cross-attention
  - 输出：每个位置的 hidden state

Step 3: Sample
  - 从 N 个 mask 位置的 hidden state 计算 logits
  - 并行采样 N 个 draft token（一次性完成）
```

### 2.3 为什么 Context KV Precompute 不在 CUDA Graph 中？

CUDA Graph 要求每次 replay 的 tensor shape 完全一致。但 context token 数量每步都不同（随着生成推进，context 越来越长，且 reject 后会回退）。因此 precompute 必须 eager 执行，只有 query-block forward（固定 1+N 个 token）才适合被 CUDA Graph 捕获。

### 2.4 DFlash vs Eagle/Medusa

| 维度 | Eagle | Medusa | DFlash |
|------|-------|--------|--------|
| Draft 方式 | 逐 token 自回归 | 多 head 并行 | Query-block 并行 |
| Forward 次数 | N 次 | 1 次 | 1 次 |
| Token 间依赖 | 完全依赖 | 无 | 无（非因果 mask） |
| KV Cache 复用 | 独立 draft KV | 共享 target | Context-KV precompute |
| CUDA Graph | 每步一个 graph | 一个 graph | Precompute + 一个 graph |

DFlash 和 Medusa 一样是并行出 N 个 token，但 DFlash 有独立的 draft 模型（不是简单的多 head），且通过 context-KV precompute 高效利用 target 模型的已有信息。

## 3. DSpark：半自回归框架（DFlash + Markov Head）

DSpark 在 DFlash 的并行 backbone 上，增加了一个轻量的**顺序 Markov Head**，以极低的代价补偿 token 间的依赖关系。

### 3.1 核心思路

DFlash 的问题：并行 draft 没有 token 间依赖，导致后面位置的接受率下降。

DSpark 的解法：**两阶段架构**
1. **Parallel Backbone**：复用 DFlash 的并行 forward，一次出 N 个位置的 base logits
2. **Sequential Markov Head**：从左到右顺序采样，每步注入前一 token 的转移偏置（Markov bias）

```
Parallel Backbone（并行，一次 forward）:
  [anchor, mask_1, mask_2, ..., mask_N]
  → [base_logits_1, base_logits_2, ..., base_logits_N]

Sequential Markov Head（顺序，N 步轻量计算）:
  Step 1: logits_1 = base_logits_1 + markov_bias(anchor_token)
          sample → draft_token_1
  Step 2: logits_2 = base_logits_2 + markov_bias(draft_token_1)
          sample → draft_token_2
  Step 3: logits_3 = base_logits_3 + markov_bias(draft_token_2)
          sample → draft_token_3
  ...
```

### 3.2 Markov Head 的低秩设计

Markov Head 的目标：给定前一个 token，产生一个 vocab 大小的 bias 向量。

朴素做法需要一个 `V × V` 的转移矩阵（V 是词表大小，如 150K），显存和计算都不可接受。

DSpark 用**低秩分解**：

```
markov_w1: Embedding(V, r)    — 把 token id 映射到 r 维（r 远小于 V，如 64-256）
markov_w2: Linear(r, V)       — 从 r 维投射回 vocab 大小的 bias

markov_bias(prev_token) = markov_w2(markov_w1(prev_token))
```

这等价于把 `V × V` 矩阵分解为 `V × r` 和 `r × V` 的乘积。计算量从 `O(V²)` 降到 `O(V × r)`，当 `r=128` 时只有原来的 ~0.1%。

### 3.3 sample_from_anchor：两种 Query 布局

DSpark 存在两种布局模式，由 `sample_from_anchor` 配置控制：

**Anchor-as-first-prediction（`sample_from_anchor=True`，DSpark 默认）**：
- N 个 query token：`[anchor, mask_1, mask_2, ..., mask_{N-1}]`
- anchor 位置本身也是预测位置（预测下一个 token）
- 每个位置 k 预测 position k+1 的 token
- 共 N 个预测

**1+N Fill-in（`sample_from_anchor=False`，DFlash 默认）**：
- 1+N 个 query token：`[anchor(bonus), mask_1, mask_2, ..., mask_N]`
- anchor 是 bonus token，不参与采样
- 只有 mask 位置参与采样
- 共 N 个预测

两种布局的区别在于 anchor 是否参与采样。DSpark 让 anchor 也参与预测，减少了一个 query 位置的开销。

### 3.4 为什么 DSpark 有效？

关键洞察：**并行 backbone 已经捕获了大部分语义信息，token 间依赖只需要一个轻量的修正**。

- Parallel backbone 的 base logits 已经很接近正确答案（类似 Medusa 的质量）
- Markov Head 只是做了一个小的偏置修正：告诉模型"前一个 token 是什么，据此调整当前的概率分布"
- 低秩设计让每步 Markov bias 的计算量极小（两次矩阵乘，维度是 `r`）

对比：
- Eagle 用完整的自回归保证依赖，但 N 次 forward 太慢
- DSpark 用低秩 Markov bias 近似依赖，一次 forward + N 步轻量计算，速度接近 Medusa 但接受率接近 Eagle

## 4. Dynamic Speculative Decoding Schedule（动态投机调度）

### 4.1 问题

传统的投机解码使用固定的 `num_speculative_tokens`（如 K=7），但实际推理中 batch size 是动态变化的：

- **低并发**（batch_size=1-4）：大模型 forward 是 memory-bound，draft 更多 token → 更大加速比
- **高并发**（batch_size=64-128）：大模型 forward 已经是 compute-bound，draft token 的验证开销反而拖累吞吐

### 4.2 方案

Dynamic SD Schedule 是一个**按 batch size 区间映射 draft 长度的静态查找表**：

```
配置示例：
  batch_size  1-16  → num_speculative_tokens = 7
  batch_size 17-64  → num_speculative_tokens = 3
  batch_size 65-128 → num_speculative_tokens = 1
```

本质是 throughput-oriented 的静态映射，不是运行时基于 logits 置信度的动态决策。Scheduler 每步根据当前 batch size 查表获取 K 值。

### 4.3 实现

实现分两步：
1. **验证和规范化**：检查配置合法性（区间不重叠、起点从 1 开始等）
2. **构建稠密查找表**：把 `[(range_start, range_end, K), ...]` 展开为 `dense_schedule[batch_size] = K`

查找表是 0(1) 的数组索引，没有运行时搜索开销。区间之间的间隙用前一个区间的 K 值填充，尾部超出最后区间的 batch size 也沿用最后的 K 值。

## 5. 方法对比总结

| 方法 | Draft 策略 | Token 间依赖 | Draft 速度 | 接受率 | 适用场景 |
|------|-----------|-------------|-----------|--------|---------|
| Eagle | 自回归 | 完全依赖 | 慢（N 次 forward） | 高 | 低延迟、低并发 |
| Medusa | 并行多 head | 无 | 快（1 次 forward） | 中（衰减快） | 吞吐优先 |
| N-gram | 模式匹配 | 无 | 零开销 | 看输入 | 重复性文本 |
| DFlash | 并行 backbone | 无 | 快（1 次 forward） | 中 | 吞吐优先 |
| DSpark | 并行 + Markov Head | 低秩偏置 | 快（1 次 forward + N 步轻量计算） | 高 | **生产部署首选** |

DSpark 是目前 DeepSeek 生产环境实际部署的方案。它在速度和质量之间找到了最佳平衡点：并行 backbone 保证速度，Markov Head 以极低代价补偿依赖，接受率接近自回归方法。

## 6. 代码路径概览

代码分为**两层架构**：

**Proposer 编排层**（Scheduler 侧，管"何时 draft"）：
- `vllm/v1/spec_decode/dflash.py` — DFlash proposer：set_inputs_first_pass 准备输入，build_model_inputs_first_pass 构建模型输入
- `vllm/v1/spec_decode/dynamic/utils.py` — Dynamic SD Schedule 的验证和查找表构建

**GPU Speculator 层**（Worker 侧，管"怎么 draft"）：
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` — DFlashSpeculator：propose() 完整流程，prepare_dflash_inputs Triton kernel 准备输入，_generate_draft 执行 forward + sample
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` — DSparkSpeculator（继承 DFlashSpeculator）：override _generate_draft，增加 _sample_sequential Markov 采样
- `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py` — DFlash 的 CUDA Graph 管理

**Draft 模型定义**：
- `vllm/model_executor/models/qwen3_dflash.py` — DFlash Qwen3 draft 模型
- `vllm/model_executor/models/qwen3_dspark.py` — DSpark Qwen3 draft 模型（含 DSparkMarkovHead）
- `vllm/model_executor/models/gemma4_dspark.py` — Gemma4 DSpark
