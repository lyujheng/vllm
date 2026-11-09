# Week 5 Day 3: Prefix Caching (APC)

## 学习目标

1. 理解 Prefix Caching 要解决的问题和应用场景
2. 掌握 vLLM 中 prefix caching 的实现原理
3. 通过实验验证 prefix caching 对 TTFT 和吞吐的影响

## 1. 问题背景

### 为什么需要 Prefix Caching？

**问题**：多个请求经常共享相同的前缀内容，导致重复计算

常见场景：
- **System prompt**：所有请求都带相同的系统提示词
- **Few-shot examples**：多个请求使用相同的示例
- **RAG**：检索到的相同文档片段
- **Multi-turn**：多轮对话中历史消息不变

**解决思路**：
- 为每个 KV Cache block 计算 hash 值
- 新请求到来时，查找已缓存的 block
- 如果 hash 匹配，直接复用 KV Cache，跳过 prefill 计算

## 2. vLLM 实现原理

### 2.1 核心配置参数

**`enable_prefix_caching`**：是否开启 prefix caching（默认 True）
- 路径：`vllm/config/cache.py:157`

**`prefix_caching_hash_algo`**：hash 算法选择（默认 xxhash）
- 路径：`vllm/config/cache.py:163`
- 可选：`xxhash`, `sha256`, `xxhash_cbor`, `sha256_cbor`

### 2.2 Hash 计算原理

#### Hash 算法（默认 xxhash）

```python
# vllm/utils/hashing.py:70-73
def xxhash(input: Any) -> bytes:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return xxhash.xxh3_128_digest(input_bytes)
```

#### Hash 输入构造

```python
# vllm/v1/core/kv_cache_utils.py:622
block_hash = hash_function((
    parent_block_hash,      # 父 block 的 hash（链式依赖）
    curr_block_token_ids,   # 当前 block 的 token ids tuple
    extra_keys              # LoRA + 多模态 + cache_salt + embeddings
))
```

#### 链式 Hash 特性

每个 block 的 hash **依赖于前一个 block**：
```
Block 0: hash([None, tokens[0:16], extra_keys])
Block 1: hash([hash_0, tokens[16:32], extra_keys])
Block 2: hash([hash_1, tokens[32:48], extra_keys])
```

**意义**：Block N 的 hash 唯一标识了**从开头到 Block N 的完整前缀**

### 2.3 完整流程

#### 启动阶段

```
EngineCore.__init__()
  ↓
if enable_prefix_caching:
  caching_hash_fn = get_hash_fn_by_name("xxhash")
  request_block_hasher = get_request_block_hasher(...)
```

#### 请求创建时（计算 hash）

```
用户发送请求
  ↓
Request.__init__()
  ↓
update_block_hashes()  ← 计算 prompt 的 block hashes
  ↓
存储在 request.block_hashes: list[BlockHash]
```

#### 调度时（查询缓存）

```
Scheduler.schedule()
  ↓
KVCacheManager.get_computed_blocks(request)
  ↓
Phase 1: 粗粒度查询（hash list 前缀匹配）
Phase 2: 细粒度查询（逐 block 验证）
  ↓
返回命中的 block 数量
  ↓
request.num_computed_tokens += (命中数 * block_size)
```

#### 执行后（写入缓存）

```
ModelRunner.execute_model()
  ↓
KVCacheManager.allocate_slots()
  ↓
cache_blocks()
  ↓
block_pool.cache_full_blocks()  ← 写入 _cache 字典
```

#### 生成新 token 后（增量更新 hash）

```
Request.append_output_token_ids(new_tokens)
  ↓
update_block_hashes()  ← 增量计算新完整 block 的 hash
```

### 2.4 代码路径清单

- **Hash 算法**：`vllm/utils/hashing.py:70-73` - xxhash 实现
- **Hash 计算**：`vllm/v1/core/kv_cache_utils.py:596-623` - hash_block_tokens
- **Hash 更新**：`vllm/v1/request.py:209, 260` - update_block_hashes 调用点
- **查询缓存**：`vllm/v1/core/block_pool.py:175-266` - get_cached_blocks
- **写入缓存**：`vllm/v1/core/block_pool.py:268-325` - cache_full_blocks
- **缓存管理**：`vllm/v1/core/single_type_kv_cache_manager.py:427-477` - cache_blocks 入口

## 3. 实验设计

### Exp1: Prefix Caching 对共享 System Prompt 的影响

**目标**：验证多个请求共享相同 system prompt 时，prefix cache 能否显著降低 TTFT

**实验配置**：
- 模型：Qwen2.5-7B
- System prompt：1024 tokens（固定，所有请求共享）
- User prompt：512 tokens（随机，每个请求不同）
- Output length：128 tokens
- 请求数：20 个
- 对比配置：
  - Config A: `--enable-prefix-caching`（开启）
  - Config B: `--no-enable-prefix-caching`（关闭）

**预期结果**：
- Config A：第 1 个请求 TTFT 正常，后续请求 TTFT 显著降低（复用 system prompt 的 KV Cache）
- Config B：所有请求 TTFT 相同（每次都重新计算 system prompt）

### Exp2: Prefix Caching 对吞吐的影响

**目标**：观察 prefix caching 对整体吞吐的提升

**实验配置**：
- 模型：Qwen2.5-7B
- 负载：
  - 共享前缀：2048 tokens（system prompt + few-shot examples）
  - 用户输入：256 tokens（随机）
  - 输出长度：128 tokens
- 请求数：30 个
- Request rate：5 req/s
- 对比配置：
  - Config A: prefix caching 开启
  - Config B: prefix caching 关闭

**观察指标**：
- TTFT P50/P99（首 token 延迟）
- Throughput（整体吞吐）
- Cache hit rate（从日志中提取）

**预期结果**：
- Config A 的 Throughput 显著高于 Config B
- Config A 的 TTFT P50/P99 显著低于 Config B（第 2 个请求开始）

## 4. 实验脚本

### 准备工作

```bash
cd /home/lvzheng/vllm
source .venv/bin/activate  # 如果有 venv
```

### Exp1 脚本框架

```bash
# TODO: 实现 exp2_prefix_cache_system_prompt.sh
# 1. 启动 vLLM server（2 种配置）
# 2. 生成共享 system prompt 的 20 个请求
# 3. 记录每个请求的 TTFT
# 4. 对比分析
```

### Exp2 脚本框架

```bash
# TODO: 实现 exp3_prefix_cache_throughput.sh
# 1. 启动 vLLM server（2 种配置）
# 2. 生成共享长前缀的 30 个请求
# 3. 以 5 req/s 发送
# 4. 记录 TTFT 和 Throughput
# 5. 对比分析
```

## 5. 学习笔记

### 2026-09-05

#### 核心发现

1. **Hash 算法**：默认 xxhash (xxh3_128)，128 位输出，非加密级但极快

2. **Hash 输入**：
   - 序列化方式：pickle.dumps
   - 输入构造：`(parent_hash, token_ids_tuple, extra_keys)`
   - Extra keys 包括：LoRA + 多模态 + cache_salt + prompt_embeds

3. **链式 Hash**：每个 block 的 hash 依赖父 block，确保整个前缀的唯一性

4. **增量计算**：每次只计算新增的完整 block，避免重复

5. **缓存查询**：Phase 1 粗粒度（hash list 前缀匹配）+ Phase 2 细粒度（逐 block 验证）

6. **Sliding Window 特殊处理**：
   - `reachable_block_mask()` 过滤窗口外的 block，不写入 prefix cache
   - `remove_skipped_blocks()` 在 allocate_slots 前释放窗口外的 block

#### 关键认知

- **Prefix Cache 优化 TTFT**：跳过已计算的前缀，直接复用 KV Cache
- **适用场景**：system prompt、few-shot、RAG、multi-turn 等共享前缀场景
- **Hash 开销很小**：xxhash 极快，对性能影响可忽略
- **存储开销**：每个 block 16 字节 hash，加上 KV Cache 本身的显存

