# Week 1 Day 3: Attention Backend 学习步骤

## 时间分配：2-3 小时

---

## Step 1: 理解整体架构（30 分钟）

### 1.1 查看关键类的定义

```bash
cd /home/lvzheng/vllm/vllm

# 查看 Backend 类
grep -n "class FlashInferBackend" vllm/v1/attention/backends/flashinfer.py -A 50

# 查看 Impl 类
grep -n "class FlashInferImpl" vllm/v1/attention/backends/flashinfer.py -A 30

# 查看 MetadataBuilder 类
grep -n "class FlashInferMetadataBuilder" vllm/v1/attention/backends/flashinfer.py -A 50
```

### 1.2 理解三层架构

```
FlashInferBackend (工厂类)
    └─ 提供静态方法（get_impl_cls, get_builder_cls）
    └─ 定义 KV Cache shape 和 stride

FlashInferMetadataBuilder (元数据构建)
    └─ build() 方法：构建 FlashInferMetadata
    └─ 初始化 Prefill/Decode Wrapper
    └─ 管理 block_tables 传递

FlashInferImpl (实际执行)
    └─ forward() 方法：执行 Attention 计算
    └─ 调用 FlashInfer Wrapper
```

**关键文件位置**：
- Backend: Line 343
- Impl: Line 1530
- MetadataBuilder: Line 620

---

## Step 2: 追踪 Block Table 的传递（45 分钟）

### 2.1 Block Table 从哪里来？

```bash
# 查看 CommonAttentionMetadata（Scheduler 传入的）
grep -n "CommonAttentionMetadata" vllm/v1/attention/backend.py -A 20

# 关键字段：
# - block_table_tensor: 所有请求的 block table 拼接
# - seq_lens: 每个请求的序列长度
# - query_start_loc: 每个请求在 batch 中的起始位置
```

**理解**：
- Scheduler 调用 `MetadataBuilder.build()` 时传入 `CommonAttentionMetadata`
- 其中包含 `block_table_tensor`（所有请求的 Block Table）

### 2.2 Block Table 如何传给 FlashInfer Wrapper？

```bash
# 查看 Wrapper 初始化（在 MetadataBuilder.build() 中）
cat vllm/v1/attention/backends/flashinfer.py | grep -n "BatchPrefillWithPagedKVCacheWrapper" -A 10
```

**核心代码**（简化）：

```python
# Line 1270-1300（大约位置）
if num_prefill_tokens > 0:
    # 构建 Prefill Wrapper
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(...)
    prefill_wrapper.plan(
        qo_indptr=query_start_loc,          # 每个请求的查询起始位置
        kv_indptr=kv_indptr,                # 每个请求的 KV 起始位置  
        kv_indices=block_table_tensor,      # Block Table！
        kv_last_page_len=last_page_len,     # 最后一个 block 的有效长度
    )

if num_decode_tokens > 0:
    # 构建 Decode Wrapper
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(...)
    decode_wrapper.plan(
        indptr=seq_start_loc,               # 每个请求的起始位置
        indices=block_table_tensor,         # Block Table！
        last_page_len=last_page_len,        # 最后一个 block 的有效长度
    )
```

**关键理解**：
- ✅ `block_table_tensor` 直接传给 Wrapper 的 `plan()` 方法
- ✅ FlashInfer Wrapper 内部保存这些信息
- ✅ Forward 时直接调用 `wrapper.run(query, kv_cache)`

### 2.3 实验：打印 Block Table

创建测试脚本：

```python
# learning/week1-attention-backend/trace_block_table.py

import torch
from vllm import LLM, SamplingParams

# 修改 FlashInferMetadataBuilder.build() 方法，临时添加打印：
# print(f"block_table_tensor: {common_attn_metadata.block_table_tensor}")
# print(f"seq_lens: {common_attn_metadata.seq_lens}")

llm = LLM(model="facebook/opt-125m", max_num_seqs=2)
prompts = ["Hello, how are you?", "What is the weather?"]

outputs = llm.generate(prompts, SamplingParams(max_tokens=5))

# 查看日志中的 block_table_tensor
```

---

## Step 3: 理解 Forward 的执行流程（45 分钟）

### 3.1 Forward 方法的核心逻辑

```bash
# 查看 forward 方法
cat vllm/v1/attention/backends/flashinfer.py | sed -n '1681,1900p'
```

**关键代码流程**：

```python
def forward(
    self,
    query: torch.Tensor,           # [num_tokens, num_heads, head_size]
    key: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
    kv_cache: torch.Tensor,        # [num_blocks, num_kv_heads, block_size, 2*head_size]
    attn_metadata: FlashInferMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    # Step 1: 分离 Decode 和 Prefill
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_prefill_tokens = attn_metadata.num_prefill_tokens
    
    decode_query = query[:num_decode_tokens]
    prefill_query = query[num_decode_tokens:]
    
    # Step 2: 调整 KV Cache 布局
    kv_cache_permute = kv_cache.permute(stride_order)  # HND 或 NHD
    k_cache, v_cache = kv_cache_permute.split(head_size, dim=-1)
    
    # Step 3: 调用 FlashInfer（Prefill）
    if num_prefill_tokens > 0:
        prefill_output = attn_metadata.prefill.wrapper.run(
            prefill_query, k_cache, v_cache
        )
        output[num_decode_tokens:] = prefill_output
    
    # Step 4: 调用 FlashInfer（Decode）
    if num_decode_tokens > 0:
        decode_output = attn_metadata.decode.wrapper.run(
            decode_query, k_cache, v_cache
        )
        output[:num_decode_tokens] = decode_output
    
    return output
```

### 3.2 关键问题解答

**Q: Wrapper 如何知道每个 token 属于哪个请求？**

A: 通过 `plan()` 时传入的 `indptr` 和 `indices`：
```python
# indptr: [0, 5, 12]  表示 req_0 用 token 0-4，req_1 用 token 5-11
# indices: [1, 2, 3, 4, 5, ...]  表示每个请求的 block table
```

**Q: FlashInfer 如何从 Block Table 读取 KV Cache？**

A: FlashInfer 内部实现（CUDA kernel）：
```cuda
// 伪代码
for each token t:
    request_id = find_request(t, indptr)
    block_table = indices[request_id]
    
    for each block b in block_table:
        physical_block = block_table[b]
        kv_data = kv_cache[physical_block]
        attention_compute(query[t], kv_data)
```

---

## Step 4: 完整流程梳理（30 分钟）

### 4.1 绘制数据流图

创建完整流程图：

```bash
cat > learning/week1-attention-backend/data_flow.txt << 'EOF'
# Attention Backend 完整数据流

┌─────────────────────────────────────────────────────────────┐
│                    Scheduler 层                              │
├─────────────────────────────────────────────────────────────┤
│  1. scheduler.schedule() 返回 scheduled_requests             │
│  2. 为每个请求分配 Block                                      │
│     kv_cache_manager.allocate_slots(req)                     │
│  3. 构建 CommonAttentionMetadata:                            │
│     - block_table_tensor: [req_0_blocks | req_1_blocks | ...]│
│     - seq_lens: [len_0, len_1, ...]                         │
│     - query_start_loc: [0, q0_len, q0_len+q1_len, ...]      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FlashInferMetadataBuilder                       │
├─────────────────────────────────────────────────────────────┤
│  4. builder.build(common_attn_metadata)                      │
│     a. 初始化 Prefill Wrapper:                               │
│        wrapper.plan(                                         │
│            qo_indptr=query_start_loc,                        │
│            kv_indices=block_table_tensor,  ◄── Block Table! │
│        )                                                     │
│     b. 初始化 Decode Wrapper:                                │
│        wrapper.plan(                                         │
│            indptr=seq_start_loc,                             │
│            indices=block_table_tensor,     ◄── Block Table! │
│        )                                                     │
│  5. 返回 FlashInferMetadata（包含已配置的 Wrapper）          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  FlashInferImpl.forward()                    │
├─────────────────────────────────────────────────────────────┤
│  6. 输入：                                                    │
│     - query: [num_tokens, num_heads, head_size]              │
│     - kv_cache: [num_blocks, num_kv_heads, block_size, 2*hs]│
│     - attn_metadata: 包含 Wrapper                            │
│                                                              │
│  7. 调用 FlashInfer:                                         │
│     prefill_output = attn_metadata.prefill.wrapper.run(      │
│         query[decode_len:], kv_cache                         │
│     )                                                        │
│     decode_output = attn_metadata.decode.wrapper.run(        │
│         query[:decode_len], kv_cache                         │
│     )                                                        │
│                                                              │
│  8. Wrapper 内部（FlashInfer CUDA kernel）:                  │
│     for each token:                                          │
│         req_id = find_request(token, indptr)                 │
│         blocks = block_table_tensor[req_id]                  │
│         for block in blocks:                                 │
│             kv_data = kv_cache[block]                        │
│             attn_output += attention(query[token], kv_data)  │
│                                                              │
│  9. 返回 output: [num_tokens, num_heads * head_size]         │
└─────────────────────────────────────────────────────────────┘

关键点总结：
1. Block Table 从 Scheduler 传入（CommonAttentionMetadata）
2. MetadataBuilder 将 Block Table 传给 FlashInfer Wrapper
3. Forward 时 Wrapper 直接使用保存的 Block Table
4. FlashInfer 在 CUDA kernel 中根据 Block Table 读取 KV Cache
EOF

cat learning/week1-attention-backend/data_flow.txt
```

### 4.2 总结核心问题的答案

更新学习笔记：

```bash
cat >> learning/week1-attention-backend/day3_notes.md << 'EOF'

## 核心问题解答

### ✅ Q1: Block Table 如何传递给 FlashInfer？

**答案**：
1. Scheduler 构建 `CommonAttentionMetadata`（包含 `block_table_tensor`）
2. `MetadataBuilder.build()` 接收 `CommonAttentionMetadata`
3. `MetadataBuilder` 调用 `wrapper.plan(kv_indices=block_table_tensor)`
4. Wrapper 内部保存 Block Table
5. `forward()` 调用 `wrapper.run()`，Wrapper 使用保存的 Block Table

### ✅ Q2: Prefill 和 Decode 的区别？

| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入 tokens | 多个（整个 prompt） | 1 个/请求 |
| Query 位置 | `query[num_decode_tokens:]` | `query[:num_decode_tokens]` |
| Wrapper | BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper |
| Attention 模式 | Causal Mask（每个 token 看之前的） | 只计算最后一个 token 的 Attention |

### ✅ Q3: FlashInfer 如何从 Block Table 读取 KV Cache？

**答案**：FlashInfer 的 CUDA kernel 内部实现：
1. 根据 `indptr` 找到每个 token 属于哪个请求
2. 根据 `indices`（Block Table）找到该请求的物理 block 列表
3. 遍历 Block Table，从 KV Cache 中读取对应的物理 block
4. 执行 Attention 计算

**关键代码**（FlashInfer 内部，不在 vLLM 代码中）：
```cuda
// FlashInfer CUDA kernel（简化）
__global__ void paged_attention_kernel(
    query, kv_cache, kv_indices, indptr, output
) {
    int token_id = blockIdx.x;
    int req_id = find_request(token_id, indptr);
    
    int* block_table = &kv_indices[req_id * max_blocks];
    
    for (int i = 0; i < num_blocks[req_id]; i++) {
        int physical_block = block_table[i];
        auto kv_data = kv_cache[physical_block];
        attn_output += compute_attention(query[token_id], kv_data);
    }
    
    output[token_id] = attn_output;
}
```

---

## Day 3 学习完成 ✅

你现在已经理解：
- ✅ FlashInfer Backend 的三层架构
- ✅ Block Table 从 Scheduler → Metadata → Wrapper 的传递
- ✅ Forward 方法如何调用 FlashInfer
- ✅ Prefill 和 Decode 的区别
- ✅ FlashInfer 如何使用 Block Table 读取 KV Cache

**下一步**：Week 1 Day 4-5（实践与总结）
EOF
```

---

## ✅ Day 3 完成标准

完成以下任务后，Day 3 学习完成：

- [ ] 阅读 `FlashInferBackend`、`FlashInferImpl`、`FlashInferMetadataBuilder` 三个类的定义
- [ ] 理解 `block_table_tensor` 的传递路径：Scheduler → CommonAttentionMetadata → MetadataBuilder → Wrapper
- [ ] 理解 `wrapper.plan()` 方法的作用（配置 Block Table）
- [ ] 理解 `forward()` 方法的执行流程（分离 Prefill/Decode，调用 Wrapper）
- [ ] 能画出完整的数据流图
- [ ] 能回答"FlashInfer 如何从 Block Table 读取 KV Cache"

**预计时间**: 2-3 小时

---

## 下一步

**Week 1 Day 4-5**: 实践与总结
- 运行完整的推理流程
- 添加日志追踪 Block Table 传递
- 对比 Prefill 和 Decode 的性能
- 完成 Week 1 总结
