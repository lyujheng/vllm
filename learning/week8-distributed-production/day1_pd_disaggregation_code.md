# Day 1: PD 分离 —— vLLM 代码实现笔记

## 1. 整体架构

PD 分离 = 两个独立的 vLLM 实例 + 一个外部 Proxy

```
用户请求 → Proxy (disagg_proxy_demo.py)
              ├→ P 端: vllm serve --kv-role kv_producer (port 8100)
              └→ D 端: vllm serve --kv-role kv_consumer (port 8200)
```

- P 端和 D 端各自有完整的 EngineCore（Scheduler + Executor + Worker）
- 两端 Scheduler **不直接通信**
- 两端 Worker 通过 connector 的传输协议通信（如 NIXL notification）

## 2. 单个 vLLM 实例的层级结构

```
AsyncLLMEngine
  └── EngineCore (CPU)
        ├── Scheduler (CPU) —— 调度决策
        │     └── KVConnector (Scheduler-side)
        └── Executor (CPU) —— 管理 Worker 进程
              ├── Worker 0 (独立进程，持有 GPU 0)
              │     └── ModelRunner + KVConnector (Worker-side)
              ├── Worker 1 (独立进程，持有 GPU 1)
              │     └── ModelRunner + KVConnector (Worker-side)
              └── ...
```

EngineCore 每一步的循环：
1. `scheduler.schedule()` → 产生 SchedulerOutput（含 kv_connector_metadata）
2. `executor.execute_model(scheduler_output)` → Worker 执行 forward + KV 传输
3. `scheduler.update_from_output(scheduler_output, model_output)` → 更新状态

## 3. KV Connector 设计

### 3.1 核心概念

- 代码位置：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- 同一个 connector 类在 P 端和 D 端都实例化，通过 `kv_role` 区分行为
  - `kv_producer`（P 端）/ `kv_consumer`（D 端）/ `kv_both`（不分离）
- 每个实例又分两个 role：`SCHEDULER`（元数据决策）和 `WORKER`（GPU 数据搬运）
- 配置：`vllm/config/kv_transfer.py`

### 3.2 Scheduler-side 方法（请求生命周期顺序）

| 方法 | P 端行为 | D 端行为 |
|------|---------|---------|
| `on_new_request` | 登记新请求 | 登记新请求 |
| `get_num_new_matched_tokens` | 通常返回 0 | 检查远端 KV 到了多少 token |
| `update_state_after_alloc` | 记录 block（后续要发出） | 记录 block（后续要写入） |
| `build_connector_meta` | 打包"要发送"的指令 | 打包"要接收"的指令 |
| `request_finished` | 触发 KV 发送，延迟释放 block | 正常释放 block |

### 3.3 Worker-side 方法

| 方法 | 作用 |
|------|------|
| `save_kv_layer` | P 端每层 forward 后，从 GPU 发出 KV 数据 |
| `wait_for_save` | 等所有层发送完成 |
| `start_load_kv` | D 端 forward 前，把收到的 KV 写入 GPU 显存 |
| `wait_for_layer_load` | 等特定层的 KV 加载完成 |
| `get_finished` | 通知哪些请求的传输已完成 |

### 3.4 两端通过 SchedulerOutput 传递 metadata

- `SchedulerOutput` 中有可选字段 `kv_connector_metadata`
- 常规推理时为 None，PD 分离时携带 connector 信息
- 这是 Scheduler → Worker 的传递（同一端内部），不是跨端通信

## 4. 数据传输方式

### 4.1 传输介质

| Connector | 传输介质 | 场景 |
|-----------|---------|------|
| example_connector | 磁盘文件（safetensors） | 教学/调试 |
| NIXL | RDMA（GPU 显存直传） | 生产首选 |
| Mooncake | RDMA + 分布式缓存 | 生产 |
| LMCache | 外部 KV Cache 服务 | KV 复用场景 |
| offloading | CPU 内存 | 单机 KV 卸载 |

### 4.2 Pull vs Push 模式

**Pull 模式：**
- Proxy 串行：先发 P 端，等完成后再发 D 端
- D 端收到请求时 KV 基本已就绪

**Push 模式：**
- Proxy 并行：同时发给 P 端和 D 端
- D 端提前收到请求，状态为 WAITING_FOR_REMOTE_KVS
- P 端完成后主动推 KV 给 D 端，D 端感知到后开始 decode
- 延迟更低

两种模式都需要握手（RDMA 场景），区别在于时序编排。

### 4.3 RDMA 握手机制

- 握手不是启动时一次性完成，而是**按需动态发生**
- 第一次遇到新的远端 engine 时，后台线程发起握手
- 交换 GPU 显存地址信息（KV Cache pool 地址、block 布局）
- 握手结果缓存在 `_remote_agents`，后续请求复用
- 有 TTL 过期机制（默认 3600s）

### 4.4 多卡并行下的 KV 传输

- 每个 GPU rank 独立发送/接收自己持有的 KV Cache
- TP：按 attention head 切分，TPMapping 处理 P/D 端 TP 不一致的情况
- PP：每个 stage 发送自己那些层的 KV
- DP：每个 rank 独立处理自己的请求

## 5. D 端 Scheduler 对 KV 等待的处理

- 请求到达 D 端 → 进入 waiting 队列
- 每轮 schedule 调用 `get_num_new_matched_tokens`
- KV 没到 → 状态置为 `WAITING_FOR_REMOTE_KVS`，跳过，调度其他请求
- KV 到了 → 恢复调度，分配 block，开始 decode
- **Scheduler 不阻塞**，始终在处理其他就绪的请求

## 6. Prefix Cache 在 PD 分离下的取舍

- 不同请求共享 prompt 前缀 → P 端本地 cache 命中，无问题
- 多轮对话 → D 端产生的 KV 不回传 P 端，P 端需重新 prefill 之前的回复部分
- 这是 PD 分离的已知代价，`kv_both` 模式可以回避（但失去分离的吞吐优势）

## 7. 关键代码文件索引

- 基类定义：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- 配置：`vllm/config/kv_transfer.py`
- 教学实现：`vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`
- NIXL 实现：`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`
- Scheduler 集成：`vllm/v1/core/sched/scheduler.py`（L700+ 处理 WAITING_FOR_REMOTE_KVS）
- Worker 集成：`vllm/v1/worker/kv_connector_model_runner_mixin.py`
- Proxy 示例：`examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`
- EngineCore 主循环：`vllm/v1/engine/core.py`（L595-614）
