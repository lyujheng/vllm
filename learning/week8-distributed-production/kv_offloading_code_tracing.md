# KV Cache Offloading 代码追踪总结

> 覆盖纯 KV Cache Offloading 和 KV Cache Offloading + PD 分离两种场景。
> 基于 vLLM v1 架构，以 SimpleCPUOffloadConnector 和 NixlConnector 为例。

---

## 1. 架构总览

### 1.1 KV Connector 框架

KV Cache 的跨存储传输（CPU offload、PD 分离、外部缓存）统一抽象为 KV Connector 框架：

```
KVConnectorBase_V1（统一接口）
├── SimpleCPUOffloadConnector   → GPU↔CPU memcpy
├── NixlConnector               → GPU↔GPU RDMA（PD 分离）
├── MultiConnector              → 包装多个子 connector（PD + offloading）
├── LMCacheConnector            → 逐层流水线传输
└── ...其他 connector
```

核心文件：
- 接口定义：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- 工厂分派：`vllm/distributed/kv_transfer/kv_connector/factory.py`
- Offloading scheduler 端：`vllm/v1/simple_kv_offload/manager.py`
- Offloading worker 端：`vllm/v1/simple_kv_offload/worker.py`
- Offloading 拷贝后端：`vllm/v1/simple_kv_offload/copy_backend.py`
- Connector bridge：`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`
- MultiConnector：`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`

### 1.2 角色分离

每个 connector 内部分为两个角色：
- **SCHEDULER 角色**：运行在 scheduler 进程，负责决策（查缓存、配对 block、跟踪状态）
- **WORKER 角色**：运行在 worker 进程，负责执行（CUDA memcpy / RDMA 传输）

### 1.3 前置依赖

KV Cache Offloading 依赖 prefix caching。没有 prefix caching 就没有 block hash，无法做缓存匹配，offloading 自动禁用。
（见 `simple_cpu_offload_connector.py:88-93`）

---

## 2. 初始化流程

### 2.1 配置驱动

用户通过 `--kv-transfer-config` 传入 JSON：

```json
// 纯 offloading
{"kv_connector": "SimpleCPUOffloadConnector", "kv_connector_extra_config": {"cpu_bytes_to_use": ...}}

// PD 分离
{"kv_connector": "NixlConnector", ...}

// PD + offloading
{"kv_connector": "MultiConnector", "kv_connector_extra_config": {
  "connectors": [
    {"kv_connector": "NixlConnector", ...},
    {"kv_connector": "SimpleCPUOffloadConnector", ...}
  ]
}}
```

### 2.2 Factory 分派

`KVConnectorFactory.create_connector()`（`factory.py:43-75`）：
1. 从 `kv_transfer_config.kv_connector` 拿到 connector 名
2. 查 `_registry` 字典（懒加载注册，底部 L152-242）
3. 动态 import 对应模块，实例化 connector

### 2.3 Scheduler 端初始化

`scheduler.py:135-158`：
- `self.connector = KVConnectorFactory.create_connector(...)` — 只要有 `kv_transfer_config` 就创建
- `self.connector` 不区分 PD / offloading，统一走多态接口
- `connector.bind_gpu_block_pool(block_pool)`（`scheduler.py:291-294`）— 延迟绑定 GPU block pool
  - Offloading connector：override，保存引用，后续用于扫描 free queue、touch block
  - PD connector（Nixl）：不 override，走基类空实现

### 2.4 MultiConnector 初始化

`multi_connector.py:169-198`：
- 按 config 中 `connectors` 数组顺序创建子 connector
- `_requests_to_connector: dict[str, int]` — per-request 绑定，记录每个请求走哪个子 connector

---

## 3. Scheduler 端：5 个调用点

### 3.1 bind_gpu_block_pool

```
scheduler.__init__() → connector.bind_gpu_block_pool(block_pool)
```

- **时机**：kv_cache_manager 构建之后
- **Offloading**：保存 GPU block pool 引用，后续 touch/scan free queue 需要
- **PD 分离**：空操作（基类默认 return）

### 3.2 get_num_new_matched_tokens

```
scheduler.schedule() → connector.get_num_new_matched_tokens(request, block_aligned_local)
```

- **时机**：prefill 调度阶段，查完本地 GPU prefix cache 之后
- **输入**：request + 本地命中的 block 对齐 token 数
- **输出**：`(ext_tokens, load_kv_async)` — 外部可提供的额外 token 数 + 是否异步加载

#### 为什么要 block 对齐？

Fine-grained prefix cache 可能命中 block 内部子片段，`num_new_local_computed_tokens` 不一定是 block_size 的倍数。
对齐后传给 connector，避免外部 load 和本地半满 block 的 CoW 竞争：
- `ext_tokens > partial_tail` → 丢弃本地尾部，用外部数据
- `ext_tokens ≤ partial_tail` → 保留本地，不从外部加载

#### 各 connector 实现差异

**Offloading**（`manager.py:242-278`）：
- 跳过本地已命中的 hash，拿剩余 hash 去 CPU block pool 做 `find_longest_cache_hit`
- 命中后 touch + pin CPU blocks，防止 LRU 淘汰

**PD 分离 - Nixl pull**（`pull_scheduler.py:34-107`）：
- 不查 hash，从 `request.kv_transfer_params` 读取远端信息
- `do_remote_prefill`：返回全部 prompt 长度 - 本地命中量
- `do_remote_decode`：返回 `remote_num_tokens` - 本地命中量
- 有 `kv_recompute_threshold`：远端增量太少时返回 0，本地重算更划算

**MultiConnector**（`multi_connector.py:385-404`）：
- 按配置顺序遍历所有子 connector
- **第一个命中的胜出**，记录到 `_requests_to_connector[req_id]`
- 任一返回 None → 整体返回 None（跳过本轮调度）
- 推荐配置顺序：PD connector 在前，offloading 在后

### 3.3 update_state_after_alloc

```
scheduler.schedule() → connector.update_state_after_alloc(request, blocks, num_external_tokens)
```

- **时机**：scheduler 为请求分配好 GPU block 之后

**Offloading**（`manager.py:282-403`）：
1. Pop 之前 pin 住的 CPU hit blocks
2. 配对 CPU→GPU block 映射（`cpu_block_ids[i] → gpu_block_ids[i]`）
3. Touch 两端 block 防止异步传输期间被淘汰（ref_cnt +1）
4. 记录 load 状态到 `_reqs_to_load`

**MultiConnector**（`multi_connector.py:406-416`）：
- 只把 `num_external_tokens` 传给胜出的子 connector，其他子 connector 收到 0

### 3.4 build_connector_meta（每轮调度末尾）

```
scheduler.schedule() 尾部 → connector.build_connector_meta(scheduler_output)
→ 打包进 scheduler_output.kv_connector_metadata
```

- **时机**：每轮调度末尾，构建 SchedulerOutput 时

**Offloading**（`manager.py:405-457`）：

Load 部分：
- 遍历 `_reqs_to_load`，找未分配 event 的请求
- 汇总 gpu/cpu block 清单，分配 `load_event` 编号
- `_load_event_to_reqs[event] = req_ids` — 一个 event 对应一批请求

Store 部分（`prepare_store_specs`）：
- **Lazy 模式**（`_prepare_lazy_store_specs`）：扫描 GPU free queue，找 ref_cnt=0 且有 hash 且 CPU 上没缓存的 block
- **Eager 模式**（`_prepare_eager_store_specs`）：跟踪每个 running 请求，找已计算完成但还没 store 过的新 block
- 两种模式在启动配置时固定，运行时不可切换

打包成 `SimpleCPUOffloadMetadata`，包含 load/store 的 block 清单和 event 编号。

### 3.5 request_finished

```
scheduler._free_request() → _connector_finished() → connector.request_finished()
```

- **时机**：请求生成完毕（`update_from_output` 检测到停止条件）
- **调用链**：EngineCore 主循环 → `scheduler.update_from_output()` → `_handle_stopped_request()` → `_free_request()` → `_connector_finished()` → `connector.request_finished()`

**Offloading**（`manager.py:763-795`）：
- **不触发 store**，只做清理工作
- 释放未消费的 CPU hit pin
- 标记 in-flight load/store 为 finished，等异步传输完成后再清理
- 始终返回 `(False, None)` — 不要求 delay_free

**PD 分离 - Nixl push**（prefill 端，`push_scheduler.py:260-293`）：
- 返回 `delay_free_blocks=True` — 阻止立即释放 GPU block
- 记录 deadline：`now + kv_lease_duration`（默认 30 秒）
- Block 在租约期间保留，供 decode 端 RDMA 拉取

---

## 4. Worker 端执行流程

### 4.1 调用链

```
EngineCore 主循环
→ model_runner.execute_model()
  → kv_connector.pre_forward(scheduler_output)    # forward 前
  → GPU forward（模型计算）
→ model_runner.sample_tokens()
  → kv_connector.post_forward(finished_req_ids)   # forward 后
  → 返回 ModelRunnerOutput（含 kv_connector_output）
→ scheduler.update_from_output()
  → _update_from_kv_xfer_finished(kv_connector_output)
```

代码位置：`vllm/v1/worker/gpu/model_runner.py`
- `pre_forward`：L1391（cudagraph 模式）/ L1412（eager 模式）
- `post_forward`：L1597（正常路径）

### 4.2 pre_forward

`kv_connector.py:61-75`（ActiveKVConnector）：
1. `bind_connector_metadata` — 绑定 scheduler 打包的 metadata
2. `start_load_kv` — 启动 load

**Offloading**：`start_load_kv` 为空实现（`worker.py:198-202`），load 推迟到 `post_forward` 的 `get_finished` 中执行，利用 GPU compute 遮盖 CPU 调度开销（~5ms）。

**PD 分离（Nixl）**：`start_load_kv` 有实际逻辑，在 forward 前就启动 RDMA 传输（延迟更大，需要尽早开始）。

### 4.3 post_forward

`kv_connector.py:77-96`（ActiveKVConnector）：
1. `wait_for_save` — 等待上一轮 save 完成
   - Offloading：空实现（store 通过 CUDA event 保证顺序）
   - PD 分离（P 端）：等逐层 `save_kv_layer` 异步发送完成
2. `get_finished` — 提交传输 + 检查完成状态
3. `build_connector_worker_meta` — 打包完成事件回报 scheduler

### 4.4 Offloading 的 get_finished（worker.py:207-267）

**提交传输（L224-246）**：
```python
# Load（CPU→GPU）：立即提交
self._backend.launch_copy(load_cpu_blocks, load_gpu_blocks, is_store=False, ...)

# Store（GPU→CPU）：先记录 compute-done event，再提交
self._store_compute_done.record(torch.cuda.current_stream())
self._backend.launch_copy(store_gpu_blocks, store_cpu_blocks, is_store=True,
                          wait_event=self._store_compute_done, ...)
```

**检查完成（L248-267）**：
- 轮询 `_load_events` / `_store_events` 中的 CUDA event
- Load 完成 → 填入 `finished_recving`，上报 scheduler
- Store 完成 → 填入 `_completed_store_events`，通过 `build_connector_worker_meta` 上报

### 4.5 DmaCopyBackend（copy_backend.py）

**架构**：后台 daemon 线程 + 生产者消费者队列

**初始化**（`init` L37-69）：
- 两个独立 CUDA stream：`load_stream`（CPU→GPU）、`store_stream`（GPU→CPU）
- 优先级设为最低（`priority=low_pri`），让 KV I/O 让位给 compute stream
- `_store_params`：`src_access_order=STREAM`（源是 GPU 活跃 KV cache，需等 compute 完成）
- `_load_params`：`src_access_order=ANY`（源是 CPU pinned memory，数据稳定）
- 启动后台线程运行 `_copy_loop`

**提交任务**（`launch_copy` L71-92）：
- 把 `(src_blocks, dst_blocks, params, is_store, event_idx, events_list, wait_event)` 丢进队列
- 主线程立即返回，不阻塞

**后台线程**（`_copy_loop` L103-130）：
1. `q.get()` 阻塞等待任务
2. `stream.wait_event(wait_event)` — store 时等 compute-done event（跨 stream 同步）
3. `copy_blocks(src_blocks, dst_blocks, params)` — 调用 `cuMemcpyBatchAsync` 批量拷贝
4. `event.record(stream)` — 记录完成事件，append 到 `events_list`

**跨 stream 同步机制**：
- CUDA event 同步发生在 GPU 硬件层面，不阻塞 CPU 线程
- `event.record(stream_A)`：stream_A 执行到此处时标记 event 完成
- `stream_B.wait_event(event)`：stream_B 暂停直到 event 被标记
- 两个 CPU 线程操作不同 CUDA stream 命令队列，互不干扰

---

## 5. 完成事件回报

### 5.1 _update_from_kv_xfer_finished（scheduler.py:2714-2741）

```
scheduler.update_from_output()
→ _update_from_kv_xfer_finished(kv_connector_output)
  → connector.update_connector_output(kv_connector_output)  # connector 内部状态更新
  → 处理 finished_recving / finished_sending                # scheduler 状态转换
```

### 5.2 connector.update_connector_output

**Offloading**（`manager.py:667-689`）：
- Load 完成：清理 `_reqs_to_load` 状态
- Store 完成：累计 worker 计数，达到 `_expected_worker_count` 后调用 `_process_store_event`
  - `_process_store_completion`：注册 CPU block hash 到索引（此后 load 路径才能查到）+ 释放两端 touch 引用

**PD 分离 - Nixl**（`base_scheduler.py:478-481`）：
- 对 `finished_recving` 中的请求停止心跳（`_stop_heartbeat`）

**MultiConnector**（`multi_connector.py:435-454`）：
- 遍历每个子 connector，传入同一个 `connector_output`
- `kv_connector_worker_meta` 按子 connector 索引拆分

### 5.3 scheduler 层面处理（scheduler.py:2728-2741）

- `finished_recving`：
  - 请求状态为 `WAITING_FOR_REMOTE_KVS` → 加入 `finished_recving_kv_req_ids`，下一轮可调度
  - 请求已 finished → 释放 block
- `finished_sending`：释放 block（P 端发送完毕，不再需要保留）

### 5.4 `finished_recving` 是通用字段

不是 PD 专属：
- PD 分离：decode 端从远端 prefill 端接收 KV 完成
- Offloading：GPU 从 CPU 接收（load）KV 完成
- 语义统一为"本端获取到了外部 KV 数据"

---

## 6. PD 分离特有机制

### 6.1 逐层传输（save_kv_layer）

`vllm/model_executor/layers/attention/kv_transfer_utils.py` 的 `@maybe_transfer_kv_layer` 装饰器：
- 每个 attention 层被装饰
- 进入 attention 前：`connector.wait_for_layer_load(layer_name)` — D 端等该层 KV 到达
- 退出 attention 后：`connector.save_kv_layer(layer_name, kv_cache, attn_metadata)` — P 端异步发送该层 KV

**Nixl 不使用此机制**（`wait_for_layer_load` 和 `save_kv_layer` 均为空实现），走全量传输。
**LMCache 使用此机制**，实现逐层流水线（P 端边算边传，D 端边收边算）。

### 6.2 租约与心跳

**Prefill 端**（`push_scheduler.py`）：
- `request_finished` 时记录 deadline = `now + kv_lease_duration`（默认 30s）
- 返回 `delay_free_blocks=True`，阻止立即释放 block
- Worker 端每轮检查过期（`base_worker.py:2146-2162`），按时间排序，oldest first

**Decode 端**（`base_scheduler.py`）：
- 每隔 `lease_duration // 6`（~5s）打包心跳进 metadata（`build_connector_meta:463-468`）
- Worker 端发送心跳给 prefill 端每个 rank（`base_worker.py:2275-2305`）

**Prefill worker 端**（`base_worker.py:2189-2208`）：
- `_handle_heartbeat`：刷新 `_reqs_to_send[req_id]` 的过期时间
- 未收到心跳 → 租约到期 → block 自动释放

---

## 7. Lazy vs Eager 模式

### 7.1 配置

通过 `kv_connector_extra_config.lazy_offload` 设置，启动时固定，运行时不可切换。

### 7.2 Lazy 模式（`_prepare_lazy_store_specs`）

- 不跟踪具体请求
- 每轮调度时扫描 GPU free queue（ref_cnt=0 的 block）
- 条件：block 有 hash 且 CPU 上没有对应缓存
- 满足条件就加入 store 清单
- **适合**：GPU 内存宽裕，block 在 free queue 停留时间长
- **风险**：高并发下 block 可能在 store 前被新请求覆盖

### 7.3 Eager 模式（`_prepare_eager_store_specs`）

- 跟踪每个 running 请求的 block 增长
- 每轮调度时找已确认计算完成但未 store 过的新 block
- 增量地把新 block 加入 store 清单
- **适合**：GPU 内存紧张，block 很快被回收
- **代价**：forward 期间占用 PCIe 带宽，GPU 和 CPU 同时持有数据（GPU 上仍被请求引用）

---

## 8. Block 生命周期与保护机制

### 8.1 ref_cnt 与 free queue

- `ref_cnt > 0`：block 被活跃请求引用，不在 free queue，不可淘汰
- `ref_cnt == 0`：无请求引用，进入 free queue，KV 数据仍在，作为 prefix cache 缓存候选
- 新请求分配 block 时从 free queue 取出复用，旧数据被覆盖

### 8.2 touch 机制

`block_pool.touch(blocks)`：
- 如果 `ref_cnt == 0`（在 free queue 中）→ 从 free queue 移除
- `ref_cnt += 1`
- 作用：保护异步传输期间的 block 不被淘汰

**使用场景**：
- `get_num_new_matched_tokens`：pin 住 CPU hit blocks
- `update_state_after_alloc`：touch GPU 和 CPU 两端 block
- `_prepare_lazy_store_specs`：touch 待 store 的 GPU block

### 8.3 传输完成后释放

`_process_store_completion`（`manager.py:721-744`）：
- 注册 CPU block hash 到索引（变得可发现）
- `cpu_block_pool.free_blocks` — 释放 CPU block 的 touch 引用
- `gpu_block_pool.free_blocks` — 释放 GPU block 的 touch 引用
- 不是真正释放 block，而是撤销 touch 加的额外 ref_cnt

---

## 9. 流程对比总图

### 纯 KV Cache Offloading

```
schedule()
├── get_computed_blocks()           → 查 GPU prefix cache
├── get_num_new_matched_tokens()    → 查 CPU block pool hash 匹配
├── allocate_slots()                → 分配 GPU block
├── update_state_after_alloc()      → 配对 CPU↔GPU block，记录 load 状态
└── build_connector_meta()          → 打包 load/store block 清单 + prepare_store_specs

execute_model()
├── pre_forward()                   → 空操作（offloading 推迟到 post_forward）
├── GPU forward
└── post_forward() → get_finished()
    ├── launch_copy(load)           → CPU→GPU memcpy（异步）
    ├── launch_copy(store)          → GPU→CPU memcpy（异步，等 compute-done）
    └── 检查完成事件               → finished_recving / completed_store_events

update_from_output()
└── _update_from_kv_xfer_finished()
    └── connector.update_connector_output()
        ├── load 完成 → 清理状态
        └── store 完成 → 注册 CPU hash + 释放引用
```

### PD 分离 + KV Cache Offloading（MultiConnector）

```
schedule()
├── get_computed_blocks()           → 查 GPU prefix cache
├── MultiConnector.get_num_new_matched_tokens()
│   ├── NixlConnector → 查 kv_transfer_params（远端信息）
│   └── OffloadConnector → 查 CPU block pool hash
│   └── 第一个命中的胜出，绑定到 _requests_to_connector
├── allocate_slots()
├── MultiConnector.update_state_after_alloc()
│   └── 只把 external tokens 传给胜出的 connector
└── MultiConnector.build_connector_meta()
    └── 两个子 connector 各自打包，合并为 MultiKVConnectorMetadata

execute_model()
├── pre_forward()
│   ├── Nixl: start_load_kv → 启动 RDMA 传输
│   └── Offloading: 空操作
├── GPU forward
└── post_forward() → get_finished()
    ├── Nixl: 检查 RDMA 传输完成
    └── Offloading: launch_copy + 检查完成

update_from_output()
└── _update_from_kv_xfer_finished()
    ├── MultiConnector.update_connector_output()
    │   ├── NixlConnector → 停止心跳
    │   └── OffloadConnector → 注册 CPU hash + 释放引用
    ├── finished_recving → WAITING_FOR_REMOTE_KVS 请求可调度
    └── finished_sending → P 端释放 block
```
