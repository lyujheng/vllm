# Day 5: DFlash + DSpark 代码追踪总结

## 1. 整体调用链

```
Scheduler (vllm/v1/core/sched/scheduler.py)
  │  dynamic_sd_lookup[batch_size] → 决定本轮 num_speculative_tokens
  ↓
Proposer (vllm/v1/spec_decode/dflash.py + llm_base_proposer.py)
  │  combine_hidden_states() → 融合 target 多层 hidden states
  │  set_inputs_first_pass() → 准备 query block 输入
  ↓
Speculator (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py)
  │  precompute_and_store_context_kv() → target hidden states 投影成 draft KV Cache
  │  prepare_dflash_inputs()           → Triton kernel 准备 [anchor, MASK, ...] 输入
  │  _generate_draft()                 → forward + sample
  │    ├─ DFlash: 并行采样
  │    └─ DSpark: _sample_sequential (Markov Head 顺序采样)
  ↓
Draft 模型 (vllm/model_executor/models/qwen3_dflash.py / qwen3_dspark.py)
     _fused_kv_weight → 所有层 KV 投影 fused 一次 GEMM
     DSparkMarkovHead → 低秩 Markov bias (W1: V×r, W2: r×V)
```

## 2. 配置入口

**文件**: `vllm/config/speculative.py`

method 自动检测逻辑（L891-898）：
- draft 模型名包含 `"dflash"` → `method = "dflash"`
- draft 模型名包含 `"dspark"` 或架构为 `Qwen3DSparkModel`/`Gemma4DSparkModel` → `method = "dspark"`

使用方式：只需指定 `--speculative-model <draft_checkpoint>`，vLLM 自动识别。

## 3. Draft 模型核心：Context-KV Precompute

**文件**: `vllm/model_executor/models/qwen3_dflash.py`

### 3.1 多层 hidden states 融合

`combine_hidden_states()`（L750-770）：
- target 模型抽取指定层（如第 8/16/24 层）的 hidden states
- concat 成 `[num_tokens, n_layers * hidden_size]`
- 通过 `fc`（Linear 层）投影回 `[num_tokens, hidden_size]`
- 哪些层被抽取由 draft 模型 hf_config 中的 `target_layer_ids` 决定，训练时固定

### 3.2 Fused KV 投影

`_build_fused_kv_buffers()`（L462）→ `_build_context_kv_buffers()`（L440-454）：
- 把所有 draft 层的 `W_K`/`W_V` 权重 concat 成一个大矩阵 `_fused_kv_weight`
- 前提：所有层的 kv_size、head_dim、num_kv_heads 必须一致（L493-500 assert 校验）
- 权重加载完成后调用一次，之后复用

`precompute_and_store_context_kv()`（L548-619）：
1. 对 context_states 做 RMSNorm（input layernorm）
2. 一次 fused GEMM：`F.linear(normed_states, _fused_kv_weight)` → 所有层的 K/V
3. 对 K 做 QKNorm（Qwen3 特性，投影后对 K 再 normalize）
4. 对 K 做 fused RoPE（所有层 batch 处理）
5. V 不做位置编码（V 只提供内容，不需要位置信息）
6. 逐层写入各自的 KV Cache（`do_kv_cache_update`）

precompute 在 CUDA Graph 之外执行（context 长度每步变化），query-block forward 在 CUDA Graph 内。

### 3.3 非因果 Attention 配置

`dflash_has_any_non_causal()`（L67-72）+ `_dflash_layer_causal()`（L58-64）：
- 逐层配置：sliding window attention 层 → 因果，全局 attention 层 → 非因果
- 由 draft 模型 hf_config 中的 `layer_types` 和 `dflash_config.causal` 决定
- 用户无需手动配置，draft checkpoint 自带

## 4. DFlash Speculator

**文件**: `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`

### 4.1 propose() 完整流程（L302-469）

1. 获取 target hidden states，存入 `self.hidden_states` buffer
2. 调 `prepare_dflash_inputs`（Triton kernel）准备：
   - input_ids: `[anchor, MASK, MASK, ..., MASK]`（1+N 个 query token）
   - positions、slot_mapping、sample_indices
3. `precompute_and_store_context_kv()`: target hidden states → draft KV Cache
4. `dispatch_cg_and_sync_dp()`: 决定是否走 CUDA Graph
5. `_generate_draft()`: forward + 并行采样

### 4.2 prepare_dflash_inputs Triton Kernel（L472-630）

每个 request 一个 program：
- anchor（query offset 0）填 `last_sampled`（上轮验证通过的最后一个 token）
- 其余位置填 `parallel_drafting_token_id`（MASK token）
- 根据 `SAMPLE_FROM_ANCHOR` 决定采样起始 offset（DSpark: 0, DFlash: 1）

## 5. DSpark Speculator

**文件**: `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`

继承 `DFlashSpeculator`，只 override 两个方法：

### 5.1 _generate_draft()（L151-169）

调 `_run_model()`（继承自 DFlash）拿到 head_hidden，然后调 `_sample_sequential()` 替代 DFlash 的并行采样。

### 5.2 _sample_sequential()（L100-149）

核心的"半自回归"采样循环：
```python
base_logits = model.compute_draft_logits(sample_hidden)  # 一次性算出所有位置的 base logits
base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

prev = anchor_token  # 初始为 anchor
for i in range(n_spec):
    markov_embed = model.markov_embed(prev)      # Embedding lookup: V → r
    bias = model.markov_bias(markov_embed)        # Linear: r → V
    logits_i = base_logits[:, i] + bias           # 加法偏置
    draft_sampled_i = sample(logits_i)            # 采样
    prev = draft_sampled_i                        # 更新 prev
```

base logits 是并行算的，采样是顺序的（依赖前一步的采样结果计算 Markov bias）。

### 5.3 sample_from_anchor 布局差异

- DFlash（`sample_from_anchor=False`）: `num_query_per_req = 1 + N`，anchor 不采样
- DSpark（`sample_from_anchor=True`）: `num_query_per_req = N`，anchor 也参与预测

体现在 scheduler 的 lookahead 设置（scheduler.py L261-270）：
- DFlash: `num_lookahead_tokens = num_spec_tokens + 1`
- DSpark: `num_lookahead_tokens = num_spec_tokens`

## 6. Markov Head

**文件**: `vllm/model_executor/models/qwen3_dspark.py`

`DSparkMarkovHead`（L35-75）：
- `markov_w1`: `nn.Embedding(vocab_size, markov_rank)` — token id → r 维
- `markov_w2`: `ParallelLMHead(draft_vocab_size, markov_rank, disable_tp=True)` — r 维 → vocab 大小
- `disable_tp=True`: 顺序执行的 head 不做 TP 分片，避免每步 all-reduce

低秩分解的意义：将 V×V 转移矩阵分解为 V×r + r×V，参数量从 O(V²) 降到 O(V×r)。

## 7. Dynamic SD Schedule

**文件**: `vllm/v1/spec_decode/dynamic/utils.py` + `vllm/v1/core/sched/scheduler.py`

构建阶段（scheduler 初始化时）：
- `build_dynamic_sd_schedule_lookup()` 将配置 `[(range_start, range_end, K), ...]` 展开为稠密数组

运行时（scheduler.py L1194-1197）：
```python
num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]
```
- `len(num_scheduled_tokens)` = 当前 batch size
- O(1) 数组索引，无运行时搜索开销

## 8. 关键设计决策总结

| 设计点 | 决策 | 原因 |
|-------|------|------|
| Context KV 来源 | 从 target hidden states 投影 | 避免 draft 重新处理 context |
| KV 投影方式 | 所有层 fused 一次 GEMM | 减少 kernel launch 开销 |
| Query block attention | 混合因果（逐层配置） | 尊重 target 模型的层类型 |
| Precompute 不进 CUDA Graph | context 长度每步变化 | CUDA Graph 要求固定 shape |
| Markov Head 不做 TP | 顺序执行，TP 需要每步 all-reduce | 开销大于收益 |
| Dynamic SD 用稠密数组 | O(1) 查找 | scheduler 热路径不能有搜索开销 |
