# Week 3 Day 1: 端到端请求全景

## 完整链路追踪

### 第一站：CLI 入口

```
vllm serve <model>
  → vllm/entrypoints/cli/serve.py: ServeSubcommand.cmd()
  → 单进程场景: uvloop.run(run_server(args))        # line 152
```

### 第二站：API Server 启动

```
vllm/entrypoints/openai/api_server.py:

run_server(args)                                      # line 751
  → run_server_worker(listen_address, sock, args)     # line 767
    → build_async_engine_client(args) as engine_client # line 778
      → AsyncLLM.from_vllm_config(vllm_config)       # line 168
    → build_and_serve(engine_client, ...)             # line 782
      → FastAPI app + uvicorn 开始监听 HTTP
```

### 第三站：AsyncLLM 初始化

```
vllm/v1/engine/async_llm.py: AsyncLLM.__init__()      # line 75

关键组件创建：
1. self.input_processor = InputProcessor(...)          # line 138
   - 负责 prompt → token_ids + 多模态处理

2. self.output_processor = OutputProcessor(...)        # line 141
   - 负责模型输出 → RequestOutput（detokenize）

3. self.engine_core = EngineCoreClient.make_async_mp_client(...)  # line 149
   - 创建 EngineCore 后台进程
   - 通过 ZMQ socket 通信（input_socket / output_socket）
```

### 第四站：请求到达（HTTP → EngineCore）

```
用户 POST /v1/completions
  → FastAPI route handler
    → engine_client.generate(prompt, sampling_params, request_id)

vllm/v1/engine/async_llm.py: generate()               # line 544
  → add_request()                                      # line 579
    → input_processor.process_inputs_async()           # line 369
      - tokenize prompt → token_ids
      - 处理多模态输入
      - 构造 EngineCoreRequest
    → output_processor.add_request(request, ...)       # line 429
      - 注册 request 到输出处理器（准备 detokenize）
    → engine_core.add_request_async(request)           # line 432
      - 序列化 request，通过 ZMQ socket 发送到 EngineCore 进程
```

### 第五站：EngineCore 进程的 busy loop

```
vllm/v1/engine/core.py: EngineCoreProc.run_busy_loop()  # line 1378

while running:
    1. _process_input_queue()                            # line 1382
       - 从 input_queue 取出请求
       - _handle_client_request() → scheduler.add_request()
       - 新请求进入 Scheduler 的 waiting 队列

    2. _process_engine_step()                            # line 1386
       - 调用 self.step_fn()（即 step() 或 step_with_batch_queue()）
       - 把输出放入 output_queue
```

### 第六站：EngineCore.step() —— 系统心跳

```
vllm/v1/engine/core.py: step()                          # line 584

def step(self):
    # 1. 调度：决定哪些请求参与本轮
    scheduler_output = self.scheduler.schedule()

    # 2. GPU 执行：异步提交给 Executor
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. 结构化输出的 grammar bitmask（如果有）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 等待 GPU 执行完成
    model_output = future.result()

    # 5. 如果需要单独采样（PP 场景）
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    # 6. 更新 Scheduler 状态（完成/继续/释放资源）
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

    return engine_core_outputs
```

### 第七站：结果返回

```
EngineCore 进程:
  output_queue.put_nowait(engine_core_outputs)
  → 通过 ZMQ output_socket 发送到 AsyncLLM 进程

AsyncLLM 进程:
  output_handler 后台 task 接收输出
  → output_processor.process_outputs()
    - sampled_token_ids → detokenize → text
    - 构造 RequestOutput
  → 放入对应请求的 RequestOutputCollector queue

generate() 协程:
  while not finished:
      out = await q.get()          # 从 queue 取结果
      yield out                     # 返回给 FastAPI handler

FastAPI handler:
  async for output in generate():
      yield f"data: {output.json()}\n\n"   # SSE 流式返回
```

---

## 架构全景图

```
┌─── API Server 进程 ───────────────────────────────────────┐
│                                                            │
│  FastAPI (uvicorn)                                         │
│    │                                                       │
│    ▼                                                       │
│  AsyncLLM                                                  │
│    ├── InputProcessor (tokenize + 多模态)                   │
│    ├── OutputProcessor (detokenize + 构造输出)              │
│    └── EngineCoreClient (ZMQ sender/receiver)              │
│              │                          ▲                   │
└──────────────│──────────────────────────│───────────────────┘
               │ ZMQ input_socket         │ ZMQ output_socket
               ▼                          │
┌─── EngineCore 进程 ───────────────────────────────────────┐
│                                                            │
│  run_busy_loop():                                          │
│    ├── _process_input_queue()                              │
│    │     → scheduler.add_request(request)                  │
│    │                                                       │
│    └── _process_engine_step()                              │
│          → step()                                          │
│              ├── scheduler.schedule() → SchedulerOutput     │
│              ├── model_executor.execute_model()             │
│              │     → Executor → Worker → ModelRunner        │
│              │       → model.forward() → logits            │
│              │       → sampling → token_ids                 │
│              └── scheduler.update_from_output()             │
│                    → 更新请求状态、释放资源                   │
│                    → EngineCoreOutputs                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. 为什么用多进程？（API Server 进程 vs EngineCore 进程）

- **隔离 GIL**：API Server 做 tokenize/detokenize 等 CPU 操作，EngineCore 跑 GPU 调度
- **解耦生命周期**：API Server 可以重启而不中断 GPU 执行
- **多 API Server 支持**：多个 API Server 进程可以连同一个 EngineCore

### 2. 为什么用 ZMQ 而不是 asyncio.Queue？

- 跨进程通信（asyncio.Queue 只能进程内）
- ZMQ 零拷贝、高吞吐
- 支持 ROUTER/DEALER 模式（多 API Server → 一个 EngineCore）

### 3. EngineCore 用 busy loop 而不是 event loop？

- `run_busy_loop()` 是同步紧凑循环，没有 asyncio 开销
- GPU 执行是阻塞等待 future.result()，不需要 async
- 延迟最低：step 之间没有任务切换

---

## 关键代码文件速查

| 层级 | 文件 | 核心方法 |
|------|------|---------|
| CLI 入口 | `vllm/entrypoints/cli/serve.py` | `ServeSubcommand.cmd()` |
| API Server | `vllm/entrypoints/openai/api_server.py` | `run_server()`, `build_async_engine_client()` |
| 异步引擎 | `vllm/v1/engine/async_llm.py` | `generate()`, `add_request()`, `_run_output_handler()` |
| IPC 客户端 | `vllm/v1/engine/core_client.py` | `MPClient`, `AsyncMPClient` |
| 引擎核心 | `vllm/v1/engine/core.py` | `EngineCore.step()`, `EngineCoreProc.run_busy_loop()` |
| 调度器 | `vllm/v1/core/sched/scheduler.py` | `schedule()`, `update_from_output()` |
| 执行器 | `vllm/v1/worker/gpu_model_runner.py` | `execute_model()` |

---

## 与 Week 2 的衔接

Week 2 学了 Scheduler 的 `schedule()` 和 `update_from_output()`。
现在知道它们的调用者是 `EngineCore.step()`，在一个独立进程的 busy loop 里反复执行。

请求进入 Scheduler 的路径：
```
HTTP → AsyncLLM.generate() → engine_core.add_request_async()
  → ZMQ → EngineCore._process_input_queue()
  → scheduler.add_request() → 进入 waiting 队列
```

---

## 遗留问题（后续 Day 深入）

1. `model_executor.execute_model()` 内部怎么到 ModelRunner？（Day 2-3）
2. ModelRunner 如何构造 input_ids / positions / attn_metadata？（Day 2-3）
3. Sampling 是在哪里做的？GPU kernel 还是 CPU？（Day 4-5）
4. CUDA Graph 在 execute_model 中怎么参与？（Day 6）
