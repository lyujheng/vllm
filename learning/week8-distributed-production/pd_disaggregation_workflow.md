# PD 分离：Prefill 端与 Decode 端协同工作流程

## 场景一：最简单情况（1P1D，磁盘传输）

**环境：** 1 张 GPU 做 Prefill，1 张 GPU 做 Decode，用 example_connector（磁盘文件传输）。

**用户请求：** "请帮我写一首诗"，prompt_token_ids = [100, 200, 300, 400]（4 个 token）

### 完整流程

```
用户 → Proxy → P 端 (prefill) → 磁盘 → D 端 (decode) → 用户
```

#### 第 1 步：Proxy 收到请求，发给 P 端

```
Proxy → HTTP POST http://P端:8100/v1/completions
        body: {prompt: "请帮我写一首诗", max_tokens: 1}
```

P 端就是一个普通的 vllm serve（kv_role=kv_producer），走正常流程：
API Server → EngineCore → Scheduler

#### 第 2 步：P 端 Scheduler 调度

Scheduler 为请求分配 KV Cache block，假设分了 block [3]（block_size=4，4 个 token 刚好 1 个 block）。

`build_connector_meta` 打包 metadata 传给 Worker：
```python
metadata = {
    request_id: "abc",
    block_ids: [3],
    is_store: True    # ← 告诉 Worker "这个请求的 KV 要存出去"
}
```

#### 第 3 步：P 端 Worker 执行 forward + 存储 KV

ModelRunner 正常跑 prefill，4 个 token 算完，KV Cache 写入 GPU 显存的 block 3。

然后 connector 的 `save_kv_layer` 被调用：
```
对每一层：
    从 GPU block 3 读出 KV 数据
    → 保存到磁盘 /tmp/{hash}/layer_0.safetensors
    → 保存到磁盘 /tmp/{hash}/layer_1.safetensors
    → ...
```

P 端完成，返回给 Proxy。

#### 第 4 步：Proxy 发请求给 D 端

Proxy 等 P 端完成后，把同一个请求完整发给 D 端：
```
Proxy → HTTP POST http://D端:8200/v1/completions
        body: {prompt: "请帮我写一首诗", max_tokens: 512}
```

注意：同一个 prompt 完整发过去，D 端知道 prompt 的所有 token。

#### 第 5 步：D 端 Scheduler 调度

Scheduler 问 connector：`get_num_new_matched_tokens(request, 0)`

connector 内部逻辑：
1. 把 prompt_token_ids 做 hash
2. 检查 `/tmp/{hash}/` 文件夹是否存在
3. 存在（P 端刚写的）→ 返回 `(4, False)`，表示 4 个 token 的 KV 可从外部加载

Scheduler 知道这 4 个 token 不用自己 prefill，分配 block [20]。

`build_connector_meta` 打包 metadata 传给 Worker：
```python
metadata = {
    request_id: "abc",
    block_ids: [20],
    is_store: False    # ← 告诉 Worker "这个请求的 KV 要从外部加载进来"
}
```

#### 第 6 步：D 端 Worker 加载 KV + decode

`start_load_kv` 被调用：
```
对每一层：
    从磁盘 /tmp/{hash}/layer_0.safetensors 读取 KV 数据
    → 写入 GPU 显存的 block 20 对应位置
```

KV 加载完毕后，ModelRunner 跳过 prefill，直接开始 decode，逐 token 生成回复。

#### 第 7 步：D 端输出返回给用户

生成的 token 流式返回：D 端 → Proxy → 用户

### 关键点总结

- P 端和 D 端是两个独立的 vllm serve 进程，各有完整的 EngineCore
- 两端 Scheduler 不直接通信
- 协同通过共享传输介质完成（此场景为磁盘文件）
- D 端收到完整 prompt，能自己算出需要多少 block
- connector 的 Scheduler-side 做决策（存/取），Worker-side 做实际数据搬运

---

## 场景二：生产环境（NIXL RDMA 传输，Push 模式）

**环境：** 节点 A（1 张 GPU）做 Prefill，节点 B（1 张 GPU）做 Decode，用 NIXL connector（RDMA 直传）。

**与场景一的核心区别：** 不经过磁盘，P 端 GPU 显存直接写入 D 端 GPU 显存。

### 启动阶段（一次性，请求到来之前）

场景一没有这个阶段。RDMA 需要知道对方的 GPU 显存地址，所以有一个额外的握手步骤。

```
P 端 Worker 启动 → 注册自己的 KV Cache pool 显存地址
D 端 Worker 启动 → 注册自己的 KV Cache pool 显存地址
```

握手不一定在启动时立即发生，而是**按需触发**：第一次需要与某个远端 engine 通信时，
后台线程通过 ZMQ 连接对方，交换显存地址信息，结果缓存复用。

握手完成后，双方知道：
- 对方 KV Cache pool 的 RDMA 地址
- 对方的 block 布局（block_size、block 数量）
- TP rank 之间的映射关系（TPMapping）

### 完整流程（Push 模式）

```
用户 → Proxy ──→ P 端 (prefill)
            └──→ D 端 (分配 block → 发 PUSH_REG → 等待 KV)
                     ↑
              RDMA write (GPU → GPU)
```

Push 模式下 Proxy 并行发请求给 P 端和 D 端。

#### 第 1 步：Proxy 收到请求，同时发给 P 端和 D 端

```
Proxy → HTTP POST http://P端:8100/v1/completions  (max_tokens: 1)
Proxy → HTTP POST http://D端:8200/v1/completions  (max_tokens: 512)
（并行发出，不等 P 端完成）
```

#### 第 2 步：D 端 Scheduler 处理请求，分配 block，发送 PUSH_REG

D 端收到请求，Scheduler 问 connector：
`get_num_new_matched_tokens(request, 0)`

connector 根据请求的 prompt 长度，知道需要 4 个 token 的 KV，返回 `(4, True)`：
- `4`：需要加载 4 个 token 的外部 KV
- `True`：KV 将通过异步方式加载

Scheduler **立刻为请求分配 block [20]**，调用 `update_state_after_alloc`。

connector 的 Scheduler-side 把 block 信息存入 `_push_pending_registrations`。

然后 `build_connector_meta` 把注册信息打包到 metadata。

Worker 的 `start_load_kv` 取出注册信息，通过 NIXL 发送 PUSH_REG 给 P 端：
```
PUSH_REG: {
    request_id: "abc",
    block_ids: [20]    ← D 端分配的 block，告诉 P 端"写到这里"
}
```

发完之后，请求进入 `WAITING_FOR_REMOTE_KVS` 状态，等待 KV 到达。

#### 第 3 步：P 端 prefill + 记录完成的 block

P 端 Scheduler 调度，Worker 执行 forward，KV Cache 写入 GPU 显存 block [3]。

prefill 完成后，`request_finished` 把 block [3] 存入 `_newly_finished_push_blocks`。
`build_connector_meta` 打包到 metadata 传给 Worker。

#### 第 4 步：P 端 Worker 匹配 PUSH_REG，发起 RDMA write

P 端 Worker 有两个信息：
- D 端发来的 PUSH_REG：请求 abc → D 端 block [20]
- 本地刚完成的 block：请求 abc → P 端 block [3]

两者匹配上，发起 RDMA write：
```
P 端 GPU block [3] → RDMA → D 端 GPU block [20]
```

利用握手时拿到的 D 端显存地址，直接写入 D 端 GPU 显存，D 端 CPU 不参与。

#### 第 5 步：D 端感知 KV 到达，开始 decode

RDMA write 完成后，D 端 connector 感知到请求 abc 的 KV 已到达。

下一轮 schedule 时，请求从 `WAITING_FOR_REMOTE_KVS` 恢复为可调度。
ModelRunner 跳过 prefill，直接 decode。

#### 第 6 步：D 端输出返回给用户

与场景一相同：D 端 → Proxy → 用户

### 与场景一的对比

| | 场景一（磁盘） | 场景二（RDMA Push） |
|--|---------------|-------------------|
| 传输介质 | 磁盘文件 | GPU 显存直传 |
| Proxy 行为 | 串行：先 P 后 D | 并行：同时发 P 和 D |
| 启动握手 | 不需要 | 需要（交换显存地址） |
| Worker 间通信 | 无（各自读写磁盘） | 有（PUSH_REG notification） |
| D 端等待方式 | 基本不等（Proxy 已串行） | WAITING_FOR_REMOTE_KVS 状态轮询 |
| 延迟 | 高（磁盘 IO + Proxy 串行） | 低（RDMA 微秒级 + 并行） |
| CPU 参与 | 高（文件读写经过 CPU） | 低（RDMA 绕过 CPU） |

## 场景三：多卡并行（TP + PD 分离）

**环境：** P 端 2 张 GPU（TP=2），D 端 2 张 GPU（TP=2），NIXL connector，Push 模式。

**与场景二的核心区别：** KV Cache 按 attention head 切分在多张 GPU 上，每个 rank 独立传输自己那部分。

### 前置知识：TP 下 KV Cache 怎么分布

假设模型有 8 个 KV head，TP=2：
```
P 端 rank 0: head 0-3 的 KV（GPU 0）
P 端 rank 1: head 4-7 的 KV（GPU 1）

D 端 rank 0: head 0-3 的 KV（GPU 0）
D 端 rank 1: head 4-7 的 KV（GPU 1）
```

每个 rank 只持有部分 head 的 KV，不是完整的。

### 启动阶段

握手是 **rank 对 rank** 进行的：
```
P rank 0 ←握手→ D rank 0   (负责 head 0-3)
P rank 1 ←握手→ D rank 1   (负责 head 4-7)
```

TPMapping 在握手时计算好，记录"我这个 D 端 rank 对应 P 端的哪个 rank"。
同构 TP（P 端和 D 端 TP 相同）下就是一一对应。

### 完整流程

#### 第 1 步：Proxy 发请求（与场景二相同）

```
Proxy → P 端 (max_tokens: 1)
Proxy → D 端 (max_tokens: 512)
```

P 端和 D 端各自只有一个 Scheduler（不管几张 GPU）。

#### 第 2 步：D 端 Scheduler 分配 block，发送 PUSH_REG

Scheduler 为请求分配 block。TP 下 Scheduler 分配的是**逻辑 block**，
所有 rank 共享同一套 block ID：
```
请求 abc → block [20]
```

每个 rank 的 Worker 各自拿到同样的 block ID，
但实际存储的是不同 head 的 KV 数据。

每个 rank 的 Worker **独立**发送 PUSH_REG 给 P 端对应的 rank：
```
D rank 0 → PUSH_REG → P rank 0:  {request_id: "abc", block_ids: [20]}
D rank 1 → PUSH_REG → P rank 1:  {request_id: "abc", block_ids: [20]}
```

#### 第 3 步：P 端 prefill

两张 GPU 并行执行 prefill（标准 TP 流程），各自产生自己负责的 head 的 KV：
```
P rank 0: head 0-3 的 KV → block [3]
P rank 1: head 4-7 的 KV → block [3]
```

`request_finished` 后，两个 rank 各自记录自己的 finished blocks。

#### 第 4 步：各 rank 独立 RDMA write

每个 rank 独立匹配 PUSH_REG，独立发起 RDMA write：
```
P rank 0 GPU block [3] (head 0-3) → RDMA → D rank 0 GPU block [20]
P rank 1 GPU block [3] (head 4-7) → RDMA → D rank 1 GPU block [20]
```

两路传输并行进行，互不干扰。

#### 第 5 步：D 端开始 decode

所有 rank 都收到 KV 后，请求恢复为可调度。
两张 GPU 并行执行 decode（标准 TP 流程）。

### 异构 TP 的情况

P 端和 D 端的 TP 可以不一样。比如 P 端 TP=2，D 端 TP=4：

```
P rank 0 (head 0-3) → D rank 0 (head 0-1) + D rank 1 (head 2-3)
P rank 1 (head 4-7) → D rank 2 (head 4-5) + D rank 3 (head 6-7)
```

TPMapping 在握手时算好这种映射关系。D 端的多个 rank 从 P 端的一个 rank 读取，
各自只取自己需要的 head 范围。

### PP（Pipeline Parallelism）的情况

PP 下不同 stage 持有不同层的 KV：
```
P PP stage 0: layer 0-15 的 KV
P PP stage 1: layer 16-31 的 KV
```

每个 PP stage 独立传输自己那些层的 KV 给 D 端对应的 PP stage。
逻辑和 TP 类似：各管各的，独立传输。

### 关键点总结

- Scheduler 只有一个（不管几张 GPU），分配逻辑 block ID
- 每个 GPU rank 有独立的 Worker + connector Worker-side 实例
- 每个 rank 独立握手、独立发 PUSH_REG、独立 RDMA 传输
- 同构 TP 下 rank 一一对应，异构 TP 通过 TPMapping 处理映射
- 多 rank 传输并行进行，不需要汇聚或协调
