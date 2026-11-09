# Day 4: 投机解码代码实现追踪

## 概述

本文档记录投机解码在vLLM v1架构中的完整代码执行路径。

---

## 1. 配置入口

### 1.1 SpeculativeConfig定义

**文件**：[vllm/config/speculative.py:83-1436](vllm/config/speculative.py#L83-L1436)

**关键配置项**：
```python
@config
class SpeculativeConfig:
    num_speculative_tokens: int  # K值，小模型一次生成几个token
    model: str | None            # draft模型路径或特殊标记（"ngram", "suffix"等）
    method: SpeculativeMethod    # 算法类型（"draft_model", "eagle", "ngram"等）
    draft_tensor_parallel_size: int | None  # draft模型的TP配置
```

**`model` vs `method`**：
- **`model`**：指向具体资源（模型路径 或 特殊标记）
  - 例子：`"facebook/opt-125m"`, `"ngram"`, `"[ngram]"`
- **`method`**：指定算法类型（如何使用这个资源）
  - 例子：`"draft_model"`, `"eagle"`, `"ngram"`, `"medusa"`
  - 通常会根据 `model` 自动推断（第686-922行）

**自动推断逻辑**（第880-922行）：
- 路径包含 `"eagle-"` → `method = "eagle"`
- 路径包含 `"eagle3"` → `method = "eagle3"`
- `model_type == "medusa"` → `method = "medusa"`
- 默认 → `method = "draft_model"`

---

## 2. Engine初始化

### 2.1 EngineCore判断是否启用投机解码

**文件**：[vllm/v1/engine/core.py:169-172](vllm/v1/engine/core.py#L169-L172)

```python
self.use_spec_decode = vllm_config.speculative_config is not None
self.check_for_draft_tokens = (
    self.use_spec_decode or vllm_config.model_config.is_diffusion
)
```

**作用**：设置标志位，后续在 `step()` 中检查是否需要处理draft tokens。

---

## 3. Worker & ModelRunner初始化

### 3.1 ModelRunner创建Proposer

**文件**：[vllm/v1/worker/gpu_model_runner.py:584-647](vllm/v1/worker/gpu_model_runner.py#L584-L647)

**根据 `method` 创建不同的Proposer**：

```python
if self.speculative_config.method == "draft_model":
    self.drafter = DraftModelProposer(self.vllm_config, self.device, self)
    
elif self.speculative_config.method == "ngram":
    self.drafter = NgramProposer(self.vllm_config)
    
elif self.speculative_config.method == "ngram_gpu":
    self.drafter = NgramProposerGPU(self.vllm_config, self.device, self)
    
elif self.speculative_config.use_eagle():
    self.drafter = EagleProposer(self.vllm_config, self.device, self)
    
elif self.speculative_config.method == "medusa":
    self.drafter = MedusaProposer(...)
    
elif self.speculative_config.method == "suffix":
    self.drafter = SuffixDecodingProposer(self.vllm_config)
```

### 3.2 DraftModelProposer加载draft模型

**文件**：[vllm/v1/spec_decode/draft_model.py:96-105](vllm/v1/spec_decode/draft_model.py#L96-L105)

```python
def _get_model(self) -> nn.Module:
    draft_vllm_config = self._create_draft_vllm_config()
    with set_model_tag("draft_model"):
        model = get_model(
            vllm_config=draft_vllm_config,
            prefix="draft_model",
        )
    return model
```

**关键**：
- 独立加载draft模型，与target模型共存于同一Worker进程
- 使用 `draft_model_config` 和 `draft_parallel_config`

---

## 4. 执行流程（核心）

### 4.1 完整调用链

```
EngineCore.step()
  ↓
Scheduler.schedule()
  ↓
Worker.execute_model(scheduler_output)
  ↓
ModelRunner.execute_model(scheduler_output)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Target模型 forward                                        │
│    [gpu_model_runner.py:4450-4456]                          │
│    model_output = self._model_forward(input_ids, positions) │
│    logits = self.model.compute_logits(hidden_states)        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 采样 target token                                         │
│    [gpu_model_runner.py:4589]                               │
│    sampler_output = self._sample(logits, spec_decode_metadata) │
│    → sampled_token_ids                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
ModelRunner.sample_tokens(grammar_output)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Draft模型生成 draft tokens                                │
│    [gpu_model_runner.py:4611-4624]                          │
│    self._draft_token_ids = self.propose_draft_token_ids(    │
│        scheduler_output,                                     │
│        sampled_token_ids,  ← target模型的输出                │
│        hidden_states,      ← target模型的隐藏状态            │
│        ...                                                   │
│    )                                                         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Target模型forward

**文件**：[vllm/v1/worker/gpu_model_runner.py:4450-4456](vllm/v1/worker/gpu_model_runner.py#L4450-L4456)

```python
with set_forward_context(...):
    model_output = self._model_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )
```

**输出**：
- `hidden_states` - 用于采样和传递给draft模型
- `logits` - 用于采样target token

### 4.3 Target模型采样

**文件**：[vllm/v1/worker/gpu_model_runner.py:4589](vllm/v1/worker/gpu_model_runner.py#L4589)

```python
with record_function_or_nullcontext("gpu_model_runner: sample"):
    sampler_output = self._sample(logits, spec_decode_metadata)
```

**输出**：
- `sampled_token_ids` - target模型采样的token，用作下一步的输入

### 4.4 Draft模型生成（关键）

**文件**：[vllm/v1/worker/gpu_model_runner.py:4611-4624](vllm/v1/worker/gpu_model_runner.py#L4611-L4624)

```python
def propose_draft_token_ids(sampled_token_ids):
    with record_function_or_nullcontext("gpu_model_runner: draft"):
        self._draft_token_ids = self.propose_draft_token_ids(
            scheduler_output,
            sampled_token_ids,  # target模型刚采样的token
            self.input_batch.sampling_metadata,
            hidden_states,      # target模型的隐藏状态
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            slot_mappings,
        )
        self._copy_draft_token_ids_to_cpu(scheduler_output)
```

**调用时机**：
- 在target模型采样完成后
- 在bookkeeping之前或之后（取决于是否使用GPU tokens）

---

## 5. Draft Token生成详细流程

### 5.1 propose_draft_token_ids分发

**文件**：[vllm/v1/worker/gpu_model_runner.py:5010-5287](vllm/v1/worker/gpu_model_runner.py#L5010-L5287)

根据 `method` 进入不同分支：

```python
def propose_draft_token_ids(self, ...):
    if spec_config.method == "ngram":
        draft_token_ids = self.drafter.propose(...)
        
    elif spec_config.method == "ngram_gpu":
        draft_token_ids = self.drafter.propose(...)
        
    elif spec_config.method == "suffix":
        draft_token_ids = self.drafter.propose(...)
        
    elif spec_config.method == "medusa":
        draft_token_ids = self.drafter.propose(
            target_hidden_states=hidden_states,  # 使用target的hidden states
            ...
        )
        
    elif spec_config.use_eagle() or spec_config.uses_draft_model():
        # EAGLE / DraftModel / DFlash / Gemma4
        draft_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,  # target模型采样的token
            ...
        )
```

### 5.2 DraftModelProposer.propose() - 自回归生成

**文件**：[vllm/v1/spec_decode/llm_base_proposer.py:502-767](vllm/v1/spec_decode/llm_base_proposer.py#L502-L767)

**核心流程**：

```python
def propose(self, num_speculative_tokens, target_token_ids, ...):
    batch_size = common_attn_metadata.batch_size()
    
    # ===== 第一个draft token =====
    # 1. 设置输入（target的token_ids, positions, hidden_states）
    num_tokens, token_indices_to_sample, common_attn_metadata = (
        self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,  # target模型采样的token
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            ...
        )
    )
    
    # 2. Draft模型 forward（第一次）
    with set_forward_context(...):
        ret_hidden_states = self.model(**model_kwargs)
        last_hidden_states = ret_hidden_states
    
    # 3. 采样第一个draft token
    draft_token_ids, draft_probs = self._sample_draft_tokens(
        last_hidden_states[token_indices_to_sample],
        sampling_metadata
    )
    
    # 如果只需要1个draft token，直接返回
    if num_speculative_tokens == 1:
        return draft_token_ids
    
    # ===== 后续draft tokens（自回归循环）=====
    draft_token_ids_list = [draft_token_ids]
    
    for token_index in range(num_speculative_tokens - 1):
        # 4. 更新输入：上一个draft token作为input_ids
        input_ids = draft_token_ids_list[-1].int()
        
        # 5. Draft模型 forward（第 token_index+2 次）
        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(input_batch_size),
            ...
        }
        with set_forward_context(...):
            ret_hidden_states = self.model(**model_kwargs)
            last_hidden_states = ret_hidden_states
        
        # 6. 采样下一个draft token
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            last_hidden_states[:batch_size],
            sampling_metadata
        )
        draft_token_ids_list.append(draft_token_ids)
    
    # 7. 返回所有draft tokens [batch_size, num_speculative_tokens]
    return torch.stack(draft_token_ids_list, dim=1)
```

**关键点**：
1. **第一次forward**：使用target的输出作为输入
2. **后续forward**：自回归生成，每次用上一个draft token作为输入
3. **循环次数**：`num_speculative_tokens - 1`（第一个在循环外生成）
4. **输出形状**：`[batch_size, num_speculative_tokens]`

---

## 6. 下一步验证流程（未完成）

### 6.1 Target模型验证（下个step）

在下一个调度step中，draft tokens会被添加到输入中：

```python
# 下个step的输入
input_ids = [sampled_token, draft_token_1, draft_token_2, ..., draft_token_K]

# Target模型一次forward验证所有位置
target_logits = target_model.forward(input_ids)  # shape: [K+1, vocab_size]
```

### 6.2 接受/拒绝采样（推测）

根据概率比较：
```python
for i, draft_token in enumerate(draft_tokens):
    P_target = target_probs[i][draft_token]
    P_draft = draft_probs[i][draft_token]
    
    if P_target >= P_draft:
        accept(draft_token)
    else:
        if random() < P_target / P_draft:
            accept(draft_token)
        else:
            reject(draft_token)
            resample_from_adjusted_distribution()
            break  # 停止验证后续tokens
```

**TODO**：找到验证和采样的具体代码实现。

---

## 7. 关键文件索引

| 功能 | 文件路径 | 关键行号 |
|------|---------|---------|
| **配置定义** | `vllm/config/speculative.py` | 83-1436 |
| **EngineCore标志位** | `vllm/v1/engine/core.py` | 169-172 |
| **创建Proposer** | `vllm/v1/worker/gpu_model_runner.py` | 584-647 |
| **Target模型forward** | `vllm/v1/worker/gpu_model_runner.py` | 4450-4456 |
| **Target模型采样** | `vllm/v1/worker/gpu_model_runner.py` | 4589 |
| **调用draft生成** | `vllm/v1/worker/gpu_model_runner.py` | 4611-4624 |
| **propose分发** | `vllm/v1/worker/gpu_model_runner.py` | 5010-5287 |
| **DraftModelProposer** | `vllm/v1/spec_decode/draft_model.py` | 19-116 |
| **SpecDecodeBaseProposer** | `vllm/v1/spec_decode/llm_base_proposer.py` | 68-799 |
| **propose核心逻辑** | `vllm/v1/spec_decode/llm_base_proposer.py` | 502-767 |

---

## 8. 总结

### 8.1 核心执行流程

```
Step N:
  1. Target模型 forward → logits
  2. Target模型 采样 → sampled_token
  3. Draft模型 生成 K 个 draft tokens（自回归）
  4. 返回 draft_token_ids 给Scheduler

Step N+1:
  1. Scheduler将 draft tokens 添加到输入
  2. Target模型验证所有位置（一次forward）
  3. 接受/拒绝采样
  4. 更新请求状态
```

### 8.2 Draft Model的自回归生成

- **第1个draft token**：使用target的输出作为上下文
- **第2-K个draft token**：串行生成，每次用上一个draft token作为输入
- **前向次数**：K次（对于draft_model方法）

### 8.3 与理论的对应

| 理论概念 | 代码实现 |
|---------|---------|
| 小模型串行生成K个tokens | `propose()` 中的 `for token_index in range(K-1)` 循环 |
| 大模型一次验证K个positions | 下一个step的target forward（input_ids包含所有draft tokens）|
| 接受/拒绝采样 | TODO：待追踪 |
| Bonus token | TODO：待追踪 |

---

**下一步**：追踪验证流程，找到接受/拒绝采样的具体实现。
