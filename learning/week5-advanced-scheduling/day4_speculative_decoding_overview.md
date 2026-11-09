# Week 5 Day 4-5: Speculative Decoding - 大小模型协同机制

## 1. 背景：为什么需要 Speculative Decoding？

### 问题：Decode 阶段低效

大模型生成文本时，decode 阶段是**逐 token 生成**的：

```
Step 1: forward(prompt)           → token_1
Step 2: forward(prompt + token_1) → token_2
Step 3: forward(prompt + token_1 + token_2) → token_3
...
```

**核心问题**：
- 每步只生成 1 个 token，但大模型权重（如 7B ~14GB）每步都要完整加载
- **瓶颈在显存带宽**（memory-bound），不在计算上
- GPU 算力利用率很低（5-10%）

示例：
- 7B 模型 forward 一次：~10ms（加载权重 9ms + 计算 1ms）
- 生成 128 个 token：128 × 10ms = **1.28 秒**

### 核心洞察

**处理 1 个 token 和处理 5 个 token 的耗时几乎相同**（瓶颈在加载权重）：

```
forward(1 token)  ≈ 10ms  (加载权重 9ms + 计算 1ms)
forward(5 tokens) ≈ 11ms  (加载权重 9ms + 计算 2ms)
```

**那能不能一次验证多个 token？**

### Speculative Decoding 的思路

**用小模型（drafter）快速猜测多个 token，大模型（verifier）一次性并行验证**

```
传统 Decode（5 步）:
  大模型 forward → token_1
  大模型 forward → token_2
  大模型 forward → token_3
  大模型 forward → token_4
  大模型 forward → token_5
  总耗时: 5 × 10ms = 50ms

Speculative Decoding（2 步）:
  小模型 forward → 猜测 [token_1, token_2, token_3, token_4, token_5]  (1ms)
  大模型 forward → 一次性验证 5 个 token                                (11ms)
  总耗时: 1ms + 11ms = 12ms  → 4x 加速
```

---

## 2. 大小模型协同工作流程

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM EngineCore                         │
│  (CPU 进程，负责调度)                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ SchedulerOutput
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                    Worker 进程 (GPU)                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ModelRunner.execute_model()                   │  │
│  │                                                        │  │
│  │  Step 1: Drafter 提议 (Propose)                       │  │
│  │  ┌──────────────────────────────────────────┐        │  │
│  │  │ Drafter (小模型/算法)                     │        │  │
│  │  │ - Draft Model (68M 小模型)                │        │  │
│  │  │ - Eagle (共享大模型权重 + 小 head)         │        │  │
│  │  │ - Medusa (大模型 + 多个 head)             │        │  │
│  │  │ - N-gram (匹配已有文本)                   │        │  │
│  │  │ - MLP Speculator                         │        │  │
│  │  └──────────────┬───────────────────────────┘        │  │
│  │                 │                                      │  │
│  │                 │ draft_token_ids: [t1, t2, t3, t4, t5]│  │
│  │                 ↓                                      │  │
│  │  Step 2: Verifier 验证 (Verify)                       │  │
│  │  ┌──────────────────────────────────────────┐        │  │
│  │  │ 大模型 (Target Model)                     │        │  │
│  │  │ forward([prompt, t1, t2, t3, t4, t5])     │        │  │
│  │  │                                           │        │  │
│  │  │ 得到 6 个位置的 logits：                   │        │  │
│  │  │ pos_0: logits_0 → 采样得 t1'              │        │  │
│  │  │ pos_1: logits_1 → 对比 t2                 │        │  │
│  │  │ pos_2: logits_2 → 对比 t3                 │        │  │
│  │  │ pos_3: logits_3 → 对比 t4                 │        │  │
│  │  │ pos_4: logits_4 → 对比 t5                 │        │  │
│  │  └──────────────┬───────────────────────────┘        │  │
│  │                 │                                      │  │
│  │                 │                                      │  │
│  │  Step 3: 接受/拒绝采样                                │  │
│  │  ┌──────────────────────────────────────────┐        │  │
│  │  │ 逐个验证 draft tokens：                    │        │  │
│  │  │ t1: P_大(t1) ≥ P_小(t1) ? ✅接受 : ❌拒绝  │        │  │
│  │  │ t2: P_大(t2) ≥ P_小(t2) ? ✅接受 : ❌拒绝  │        │  │
│  │  │ t3: 拒绝 → 停止，用 t3' 代替               │        │  │
│  │  │                                           │        │  │
│  │  │ 结果：接受 [t1, t2]，添加 t3'              │        │  │
│  │  │ → 一次大模型 forward 生成 3 个 token       │        │  │
│  │  └──────────────────────────────────────────┘        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 详细执行流程

### Phase 1: Drafter 提议 (Propose)

**位置**：`ModelRunner.execute_model()` 内部

```python
# vllm/v1/worker/gpu_model_runner.py

# Step 1: Drafter 生成候选 tokens
if self.drafter is not None:
    # 调用 drafter.propose()
    draft_output = self.drafter.propose(
        input_ids=current_tokens,        # 当前已生成的 tokens
        kv_cache=kv_cache,               # KV Cache
        num_draft_tokens=self.num_spec_tokens,  # 猜测数量 K（如 5）
    )
    # 得到 draft_token_ids: [t1, t2, t3, t4, t5]
```

**不同 Drafter 的实现**：

#### Draft Model（独立小模型）
```python
# 单独的小模型（如 68M 参数）
draft_output = draft_model.forward(input_ids)
draft_token_ids = sample(draft_output.logits)
# 优点：猜测质量较高
# 缺点：占用额外显存
```

#### Eagle（共享大模型权重）
```python
# 复用大模型的浅层特征，加小 head 预测
hidden_states = target_model.get_hidden_states(input_ids, layers=[0:12])
draft_logits = eagle_head(hidden_states)
draft_token_ids = sample(draft_logits)
# 优点：共享权重，猜测质量很高
# 缺点：需要训练 eagle_head
```

#### N-gram（零成本）
```python
# 从历史文本中匹配 n-gram 模式
draft_token_ids = match_ngram_from_history(current_tokens, n=4)
# 优点：零显存开销，纯 CPU 计算
# 缺点：只适合重复性强的场景
```

### Phase 2: Verifier 验证 (Verify)

**大模型一次性处理所有候选 tokens**：

```python
# 构造输入：原始 tokens + draft tokens
verify_input = [prompt_tokens, t1, t2, t3, t4, t5]

# 大模型 forward，得到所有位置的 logits
logits = target_model.forward(verify_input)
# logits.shape: [batch, 6, vocab_size]

# 每个位置采样/验证
for i in range(len(draft_token_ids)):
    prob_target = softmax(logits[i])
    prob_draft = draft_probs[i]  # drafter 给的概率
    
    # 接受/拒绝判断（简化版）
    if prob_target[draft_token_ids[i]] >= prob_draft[draft_token_ids[i]]:
        accept_token(draft_token_ids[i])  # ✅ 接受
    else:
        reject_and_resample(logits[i])    # ❌ 拒绝，用大模型的结果
        break  # 停止验证后续 tokens
```

### Phase 3: 更新状态

```python
# 更新 num_computed_tokens（已计算的 token 数）
request.num_computed_tokens += num_accepted_tokens + 1  # +1 是修正的 token

# 更新 KV Cache（只保留接受的 tokens）
kv_cache.trim_to(num_accepted_tokens)

# 返回结果给 Scheduler
return ModelRunnerOutput(
    sampled_token_ids=[t1, t2, t3'],  # 接受的 + 修正的
    num_accepted_draft_tokens=2,      # 接受了 2 个
)
```

---

## 4. 关键特性：无损保证

Speculative Decoding **不改变大模型的输出分布**。通过特殊的采样算法确保：

**定理**：最终生成的文本序列与直接用大模型生成的序列，概率分布完全相同。

```
P_speculative(token_sequence) = P_target_only(token_sequence)
```

**实现原理**（接受/拒绝采样）：

```python
# 简化版伪代码
def verify_token(draft_token, prob_target, prob_draft):
    alpha = prob_target[draft_token] / prob_draft[draft_token]
    
    if random.uniform(0, 1) < alpha:
        return draft_token  # 接受
    else:
        # 拒绝，从修正后的分布重新采样
        adjusted_prob = max(0, prob_target - prob_draft)
        return sample(adjusted_prob / adjusted_prob.sum())
```

---

## 5. 性能分析

### 加速比公式

假设每次猜 K 个 token，接受率为 α：

- **平均每次大模型 forward 生成的 token 数**：`1 / (1 - α * K)`
- **加速比**：`(1 / (1 - α * K)) / (1 + overhead)`

其中 `overhead` 是 drafter 的相对开销。

**示例**：
- K=5，α=80%，overhead=10%
- 平均生成：1 / (1 - 0.8×5) = 5 tokens
- 加速比：5 / 1.1 ≈ **4.5x**

### 适用场景

| 场景 | 加速效果 | 原因 |
|------|---------|------|
| **代码生成** | ⭐⭐⭐⭐⭐ | 语法固定，重复性强，α 高 |
| **翻译** | ⭐⭐⭐⭐ | 对应关系明确，α 较高 |
| **摘要/问答** | ⭐⭐⭐ | 内容较确定，α 中等 |
| **创意写作** | ⭐⭐ | 不确定性大，α 低 |
| **高并发服务** | ⭐ | GPU 已饱和，drafter 成负担 |

---

## 6. vLLM 中的部署方式

### 大小模型在同一个 Worker 进程中

```
Worker 进程 (GPU 0):
  ├─ Drafter (小模型 or Eagle head)  ← 占用部分显存
  └─ Target Model (大模型)            ← 占用大部分显存
```

**内存分配**：
- 大模型权重：~14GB (7B FP16)
- 小模型权重：~136MB (68M FP16) 或 Eagle head ~50MB
- KV Cache：剩余显存

**执行顺序**：严格串行
```
1. Drafter forward (小模型)   ← GPU 计算
2. Target forward (大模型)    ← GPU 计算
3. Verify & Sample           ← CPU/GPU 混合
```

**为什么不并行？**
- 需要等 drafter 的结果才能构造 target 的输入
- 即使用多 stream，也是串行依赖关系

---

## 7. 数据传递路径

```
Scheduler (CPU)
  ↓ scheduled_spec_decode_tokens (每个请求的 draft tokens)
  ↓
Worker.execute_model()
  ↓
ModelRunner._prepare_inputs()
  ↓ 构造 spec_decode_metadata
  ↓
ModelRunner._execute_drafter()  (如果需要重新生成 draft tokens)
  ↓ draft_token_ids
  ↓
ModelRunner._model_forward()  (大模型验证)
  ↓ logits
  ↓
ModelRunner._sample_with_verification()
  ↓ accepted tokens + rejection sampling
  ↓
ModelRunnerOutput
  ↓
Scheduler.update() (更新 num_computed_tokens)
```

---

## 8. 代码路径清单

### Drafter 实现
- `vllm/v1/spec_decode/draft_model.py` — 独立小模型
- `vllm/v1/spec_decode/eagle.py` — Eagle 方法
- `vllm/v1/spec_decode/medusa.py` — Medusa 方法
- `vllm/v1/spec_decode/ngram_proposer_gpu.py` — N-gram 方法

### 验证逻辑
- `vllm/v1/worker/gpu_model_runner.py:4166` — `execute_model()` 主流程
- `vllm/v1/sample/ops/topk_topp_sampler.py` — 采样逻辑

### Scheduler 端
- `vllm/v1/core/sched/scheduler.py:468` — `scheduled_spec_decode_tokens` 管理
- `vllm/v1/core/sched/scheduler.py:1666` — 验证结果更新

---

## 9. 下一步：代码跟踪

现在可以按以下顺序跟踪代码：

1. **Scheduler 端**：`scheduled_spec_decode_tokens` 如何传递
2. **Worker 端**：`execute_model()` 中如何调用 drafter
3. **Drafter 实现**：以 N-gram 为例，看最简单的 drafter
4. **验证逻辑**：大模型如何一次性验证多个 tokens
5. **采样算法**：接受/拒绝采样的实现

要开始代码跟踪吗？
