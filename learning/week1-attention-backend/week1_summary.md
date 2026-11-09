# Week 1 总结：Attention Backend 深入理解

> 学习时间：2026-08-06 ~ 2026-08-13  
> 核心目标：理解 vLLM Attention Backend 的架构与工作原理

---

## 📚 本周学习内容

### Day 1-2：基础概念与架构理解
- vLLM 整体架构概览
- PagedAttention 原理
- Attention Backend 的作用与设计

### Day 3：FlashInfer Backend 源码阅读
- FlashInfer 三层架构（Backend / MetadataBuilder / Impl）
- Block Table 传递路径
- `forward()` 方法完整流程

### Day 4-5：实践实验
- 运行 OPT-125M 模型推理
- 观察 Prefill / Decode 过程
- 理解 SamplingParams 参数作用

---

## 🎯 核心知识点总结

### 1. PagedAttention 核心思想

```
传统方法：
  每个请求分配连续的 KV Cache 空间
  问题：内存碎片、浪费严重

PagedAttention：
  KV Cache 按固定大小的 block 分配（如 16 tokens/block）
  逻辑地址（Block Table）→ 物理地址（实际 GPU 内存）
  
优势：
  ✅ 减少内存碎片
  ✅ 支持动态增长（按需分配 block）
  ✅ 提高 GPU 利用率
```

---

### 2. FlashInfer 三层架构

```
┌─────────────────────────────────────────┐
│   FlashInferBackend (工厂类)             │
│   - get_impl_cls()                       │
│   - get_builder_cls()                    │
│   - get_kv_cache_shape()                 │
│   - get_kv_cache_stride_order()          │
│   - get_supported_kernel_block_sizes()   │
└─────────────────┬───────────────────────┘
                  │ 提供静态配置和类引用
                  ▼
┌─────────────────────────────────────────┐
│   FlashInferMetadataBuilder (准备层)     │
│   - build(common_attn_metadata)          │
│   - 初始化 Prefill/Decode Wrapper        │
│   - wrapper.plan(kv_indices=block_table) │
│   - 返回 FlashInferMetadata              │
└─────────────────┬───────────────────────┘
                  │ 传递配置好的 Metadata
                  ▼
┌─────────────────────────────────────────┐
│   FlashInferImpl (执行层)                │
│   - forward(query, key, value, kv_cache) │
│   - 调用 wrapper.run() 执行计算          │
│   - 处理 Prefill 和 Decode              │
└─────────────────────────────────────────┘
```

**关键理解**：
- **Backend**：静态配置（不参与实际计算）
- **MetadataBuilder**：运行时准备（每个 batch 调用一次）
- **Impl**：实际执行 Attention（每个 forward 调用一次）

---

### 3. Block Table 传递路径

```
┌─────────────────────────────────────────────────────────┐
│                    Scheduler 层                          │
│  1. scheduler.schedule() 返回 scheduled_requests         │
│  2. kv_cache_manager.allocate_slots(req)                 │
│  3. 构建 CommonAttentionMetadata:                        │
│     - block_table_tensor: [req_0 | req_1 | ...]         │
│     - seq_lens: [len_0, len_1, ...]                     │
│     - query_start_loc: [0, q0, q0+q1, ...]              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              FlashInferMetadataBuilder                   │
│  4. builder.build(common_attn_metadata)                  │
│     a. Prefill Wrapper.plan(                             │
│            kv_indices=block_table_tensor  ◄─ Block Table│
│        )                                                 │
│     b. Decode Wrapper.plan(                              │
│            indices=block_table_tensor     ◄─ Block Table│
│        )                                                 │
│  5. 返回 FlashInferMetadata（含已配置 Wrapper）          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  FlashInferImpl.forward()                │
│  6. 输入：query, kv_cache, attn_metadata                 │
│  7. 调用 FlashInfer:                                     │
│     wrapper.run(query, kv_cache)                         │
│                                                          │
│  8. Wrapper 内部（CUDA kernel）:                         │
│     for each token:                                      │
│         req_id = find_request(token, indptr)             │
│         blocks = block_table_tensor[req_id]              │
│         for block in blocks:                             │
│             kv_data = kv_cache[block]  ◄─ 物理索引      │
│             attn_output += attention(query, kv_data)     │
└─────────────────────────────────────────────────────────┘
```

**核心要点**：
- Block Table 从 Scheduler 传入
- MetadataBuilder 调用 `wrapper.plan()` 配置 Block Table
- Forward 时 Wrapper 使用保存的 Block Table 读取 KV Cache

---

### 4. forward() 方法关键步骤

文件：`vllm/v1/attention/backends/flashinfer.py` Line 1681-end

#### Step 1：裁剪 CUDA Graph Padding

```python
# Line 1789-1794
query = query[:num_actual_tokens]
key = key[:num_actual_tokens]
value = value[:num_actual_tokens]
output = output[:num_actual_tokens]
```

**原因**：CUDA Graph 要求固定大小，所以输入会被 padding，这里裁剪回实际大小。

---

#### Step 2：分离 Prefill 和 Decode

```python
# Line 1814-1817
num_decode_tokens = attn_metadata.num_decode_tokens
num_prefill_tokens = attn_metadata.num_prefill_tokens

decode_query = query[:num_decode_tokens]
prefill_query = query[num_decode_tokens:]
```

**区别**：
| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入 tokens | 多个（整个 prompt） | 1 个/请求（或多个，spec decoding） |
| Attention 模式 | Causal Mask | 只计算最后一个 token |
| Wrapper | BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper |

---

#### Step 3：调整 KV Cache 布局

```python
# Line 1819-1820
kv_cache_permute = kv_cache.permute(*stride_order)
```

**stride_order 两种模式**：
- **NHD** `(0,2,1,3)`：按 (num_blocks, block_size, num_heads, dim) 排列
- **HND** `(0,1,2,3)`：按 (num_blocks, num_heads, block_size, dim) 排列

---

#### Step 4：分离 K 和 V

```python
# Line 1837-1855
if self.is_kvcache_nvfp4:
    # NVFP4：K/V 存储在不同的 head 维度
    k_cache, v_cache = kv_cache.split(self.num_kv_heads, dim=1)
    # 进一步分离数据和缩放因子（量化）
    k_data, k_sf = nvfp4_split_data_scale(k_cache)
    v_data, v_sf = nvfp4_split_data_scale(v_cache)
else:
    # 普通格式：K+V 拼接在最后一维
    kv_cache_tuple = kv_cache_permute.split(head_size, dim=-1)
    # 结果：(K_cache, V_cache)
```

**两种存储格式**：
- **普通格式**：`[..., 2*head_size]` → K+V 拼接
- **NVFP4 格式**：`[..., 2*num_kv_heads, ...]` → K 和 V 的 heads 拼接

---

#### Step 5：调用 FlashInfer Wrapper

```python
# Prefill
if num_prefill_tokens > 0:
    prefill_output = attn_metadata.prefill.wrapper.run(
        prefill_query, k_cache, v_cache
    )

# Decode
if num_decode_tokens > 0:
    decode_output = attn_metadata.decode.wrapper.run(
        decode_query, k_cache, v_cache
    )
```

**Wrapper 内部**（FlashInfer CUDA kernel）：
1. 根据 `indptr` 找到每个 token 属于哪个请求
2. 根据 `indices`（Block Table）找到该请求的物理 blocks
3. 从 KV Cache 读取对应的物理 block
4. 执行 Attention 计算

---

### 5. FlashInfer vs FlashAttention

| 维度 | FlashAttention | FlashInfer |
|------|---------------|------------|
| **项目关系** | 独立项目（Tri Dao） | 独立项目（Apache TVM） |
| **核心算法** | Tiling + Online Softmax | 相同的核心思想 |
| **Paged KV Cache** | FlashAttention-2 后期加入 | 原生设计支持 |
| **Tile 划分** | tile_size 独立于 page_size | tile_size = page_size（对齐） |
| **API 设计** | 单阶段调用 | Plan/Run 两阶段 |
| **Decode 优化** | 通用 kernel | 专门的 Decode kernel |

**vLLM 的选择**：
- v1 API 默认使用 **FlashInfer**（更适合 Paged KV Cache）
- v0 API 支持多个 backend（FlashAttention, FlashInfer, xFormers 等）

---

### 6. KV Cache 存储格式

#### Shape

```python
kv_cache.shape = (num_blocks, num_kv_heads, block_size, 2 * head_size)
                  ↑           ↑              ↑            ↑
                  物理blocks  KV头数         每block tokens  K+V拼接
```

**示例**：
```
num_blocks = 1000        # 1000 个物理 block
num_kv_heads = 32        # 32 个 KV heads
block_size = 16          # 每个 block 存 16 个 tokens
head_size = 128          # 每个 head 的维度
→ shape = (1000, 32, 16, 256)
```

---

#### K+V Interleaved 存储

```
内存布局（最后一维）：
[K_dim0, K_dim1, ..., K_dim127, V_dim0, V_dim1, ..., V_dim127]
 ←────── head_size=128 ────→   ←────── head_size=128 ────→

优势：
  ✅ K 和 V 在同一个 HBM 访问中加载（连续）
  ✅ 在 GPU SRAM 中分离（zero-copy split）
```

---

#### Stride Order

**NHD** `(0,2,1,3)` vs **HND** `(0,1,2,3)`：

```
NHD (0,2,1,3):
  kv_cache[b][h][t][d] → 内存地址顺序：b → t → h → d
  适合：逐 token 处理（先遍历所有 heads 的同一个 token）

HND (0,1,2,3):
  kv_cache[b][h][t][d] → 内存地址顺序：b → h → t → d
  适合：逐 head 处理（先处理一个 head 的所有 tokens）
```

---

### 7. CUDA Graph Padding

#### 为什么需要 Padding？

```
CUDA Graph 要求：
  - 每次调用的 tensor shape 必须完全一致
  - 预先捕获计算图（capture），运行时直接 replay

vLLM 的做法：
  - 预定义约 50 个大小的 CUDA Graph（1,2,4,8,16,24,32,...,512）
  - 输入 batch_size 向上 padding 到最近的捕获大小
  
示例：
  batch_size = 19 → padding 到 24
  batch_size = 127 → padding 到 128
```

#### 配置

文件：`vllm/config/compilation.py` Line 692-707

```python
cudagraph_capture_sizes = [1,2,4] + list(range(8, 256, 8)) + list(range(256, max_size+1, 16))
# 结果：[1,2,4, 8,16,24,32,...,248, 256,272,288,...,512]
```

#### Padding 与 Block Size 的关系

**两者是正交的**：
- **CUDA Graph Padding**：控制 batch 中的 token 总数
- **Block Size**：控制 KV Cache 的分配粒度

```
batch_size padding 到 32 tokens
block_size = 16

不要求：32 必须是 16 的倍数
原因：padding 只影响输入 tensor 大小，不影响 block 分配
```

---

### 8. Speculative Decoding（简介）

代码注释：Line 1814-1817

```python
# When using spec decoding, num_decodes can be < num_decode_tokens
# because some decode requests may have more than one query token.
```

**概念**：
```
传统 Decode：每个请求每次生成 1 个 token
  Request 1: [q] → 生成 1 token
  Request 2: [q] → 生成 1 token
  num_decode_tokens = 2

Speculative Decoding：用小模型预测多个 token，大模型验证
  Request 1: [q1, q2, q3] → 验证 3 个 speculative tokens
  num_decode_tokens = 3
  num_decodes = 1（只有 1 个请求）
```

**Week 1 结论**：了解概念即可，Week 9+ 会深入学习。

---

## 🧪 实践经验

### 运行环境配置

**安装方式**（遇到的问题与解决）：
1. ❌ 源码编译失败（网络问题、缺少预编译 wheel）
2. ✅ 使用 `uv pip install vllm --system`（PyPI 稳定版）

**注意事项**：
- 在 vLLM 源码目录下运行 Python 会优先导入本地源码（未编译）
- 解决方案：切换到其他目录（如 `/tmp`）再运行脚本

---

### 推理测试观察

**模型**：OPT-125M（125M 参数）

**观察到的现象**：
1. ✅ vLLM 正常加载模型、分配 KV Cache blocks
2. ❌ 生成文本重复（temperature=0 + 小模型）
3. ❌ 不会自动生成 EOS token（模型太小，学不好）
4. ✅ 通过 `repetition_penalty=1.2` 和 `stop=["\n\n", "."]` 缓解

**SamplingParams 参数理解**：
- `temperature`：控制随机性（0=确定性，>1 更随机）
- `top_k`：只保留概率最高的 k 个 token
- `top_p`：nucleus sampling，保留累积概率达到 p 的 token 集合
- `repetition_penalty`：惩罚已出现的 token，减少重复
- `max_tokens`：限制生成的 token 数（不包括 prompt）
- `stop`：遇到指定字符串停止生成

---

## ❓ 解决的关键问题

### Q1: Block Table 如何传递给 FlashInfer？

**答**：
1. Scheduler 构建 `CommonAttentionMetadata`（包含 `block_table_tensor`）
2. `MetadataBuilder.build()` 接收并调用 `wrapper.plan(kv_indices=block_table_tensor)`
3. Wrapper 内部保存 Block Table
4. `forward()` 调用 `wrapper.run()`，Wrapper 使用保存的 Block Table

---

### Q2: Prefill 和 Decode 有何区别？

**答**：
- **Prefill**：处理整个 prompt，每个 token 需要看之前的所有 tokens（Causal Mask）
- **Decode**：每次只处理新生成的 1 个 token，需要看整个历史（包括 prompt）

---

### Q3: FlashInfer 如何从 Block Table 读取 KV Cache？

**答**：FlashInfer 的 CUDA kernel 内部：
1. 根据 `indptr` 找到每个 token 属于哪个请求
2. 根据 `indices`（Block Table）找到该请求的物理 blocks
3. 遍历 Block Table，从 `kv_cache[physical_block]` 读取数据
4. 执行 Attention 计算

---

### Q4: 为什么小模型不自动生成 EOS？

**答**：
1. **模型太小**（125M）：学不好何时该结束
2. **Greedy decoding** (temperature=0)：EOS 概率低，永远选不到
3. **repetition_penalty 副作用**：可能进一步降低 EOS 概率

解决方案：
- 使用更大的模型（7B+）
- 添加 `stop` 参数显式停止
- 提高 `temperature` 增加随机性

---

## 📁 创建的文件

```
learning/week1-attention-backend/
├── learning_steps.md     # Day 3 详细学习步骤
├── test_inference.py     # Day 4-5 推理测试脚本
├── doc.md               # vLLM 安装文档（临时参考）
└── week1_summary.md     # 本文件（Week 1 总结）
```

---

## 🎓 Week 1 学习成果

### 掌握的核心概念
✅ PagedAttention 原理与优势  
✅ FlashInfer Backend 三层架构  
✅ Block Table 传递路径  
✅ forward() 方法完整流程  
✅ KV Cache 存储格式（Shape, Stride, K+V Interleaved）  
✅ CUDA Graph Padding 机制  
✅ Prefill vs Decode 的区别  
✅ SamplingParams 参数作用  

### 阅读的核心代码
✅ `vllm/v1/attention/backends/flashinfer.py`  
  - FlashInferBackend (Line 343)  
  - FlashInferMetadataBuilder (Line 620)  
  - FlashInferImpl (Line 1530)  
  - forward() (Line 1681-end)  

✅ `vllm/v1/cudagraph_dispatcher.py`  
  - CUDA Graph padding 逻辑  

✅ `vllm/config/compilation.py`  
  - cudagraph_capture_sizes 配置  

---

## 🚀 Week 2 预告

### 学习目标
- 理解 KV Cache Manager 如何分配/释放 blocks
- 理解 Scheduler 如何调度请求、构建 batch
- 在源码中添加日志，观察 Block Table 传递过程

### 核心文件
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/scheduler.py`
- `vllm/v1/attention/backend.py`（CommonAttentionMetadata 定义）

### 实践任务
- 添加日志输出 Block Table
- 观察多请求并发时的 block 分配
- 理解 Continuous Batching 的实现

---

## 💡 学习建议

1. **循序渐进**：Week 1 打好基础，Week 2-8 逐步深入
2. **实践验证**：理论理解后一定要运行代码验证
3. **画图梳理**：复杂流程用流程图梳理（如 Block Table 传递）
4. **记录问题**：遇到不理解的先记录，后续学习中可能会豁然开朗
5. **不追求完美**：小模型生成质量差是正常的，学习重点是理解架构

---

**Week 1 完成时间**：2026-08-13  
**下一步**：Week 2 - KV Cache 管理深入
