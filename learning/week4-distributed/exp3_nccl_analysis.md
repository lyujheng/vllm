# 实验3笔记：NCCL 通信日志分析（TP=4）

> 实验时间：2026-09-01
> 配置：Qwen2.5-7B, TP=4, gpu_memory_utilization=0.8
> 环境：8× H800 80GB, NCCL 2.30.7 + CUDA 13.3, vLLM 0.27.1

---

## 1. Worker 启动流程

### 1.1 Executor 并行 spawn Worker

EngineCore（pid=26477）启动后，MultiprocExecutor 并行 spawn 4 个 Worker 进程：

```
10:19:14  Worker pid=26744  rank=0  local_rank=0  cudaDev=0
10:19:15  Worker pid=26747  rank=3  local_rank=3  cudaDev=3
10:19:16  Worker pid=26745  rank=1  local_rank=1  cudaDev=1
10:19:17  Worker pid=26746  rank=2  local_rank=2  cudaDev=2
```

关键观察：
- 启动顺序 0→3→1→2，不等于 rank 顺序，因为是并行 spawn，由 OS 调度决定
- rank == local_rank：单机场景，全局编号和本机编号一致
- cudaDev == rank：每个 Worker 绑定对应 GPU

### 1.2 TCP Store 互相发现

```
distributed_init_method=tcp://127.0.0.1:59371 backend=nccl
```

- PyTorch 内置的 TCP Store 机制：Rank 0 在 59371 端口启动轻量 KV 存储服务
- 其他 Worker 连上来注册自己的信息
- barrier 同步确保所有 Worker 到齐后才继续
- TCP Store 只在初始化阶段使用，后续数据传输走 NCCL

### 1.3 NCCL Bootstrap

```
Bootstrap timings total 0.025772 (create 0.000103, send 0.024065, recv 0.000530, ring 0.000085)
```

TCP Store 到齐后，每个 rank 调用 `ncclCommInitRank`，NCCL 内部做一轮信息交换：
- 每个 rank 发送自己的 GPU 拓扑信息（busId、NVLink、P2P 能力）
- 收集完成后开始拓扑分析和通道建立

---

## 2. NCCL 拓扑发现

### 2.1 插件探测

NCCL 启动时尝试加载可选插件，大部分未安装（INFO 级别，不影响功能）：
- `libnccl-env.so` — 环境配置插件，未找到
- `libnccl-net.so` — 网络插件，未找到
- `libnccl-rma.so` — RMA 插件，未找到
- `libnccl-gin.so` — GIN 插件，未找到
- IB（InfiniBand）— `No device found`，容器内未挂 IB 设备（单机不需要）

最终选择：`Using network Socket`（仅用于 bootstrap，实际数据走 P2P）

### 2.2 P2P 连接

```
Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
```

4 卡之间全部支持 P2P 直通，所有 channel 走 `P2P/CUMEM`（GPU 直接内存访问）。

---

## 3. 三种 AllReduce 算法

NCCL 初始化时同时建立三种 AllReduce 拓扑，运行时按数据大小自动选择：

### 3.1 Ring AllReduce

```
Channel 00/16 : 0 1 2 3
Channel 01/16 : 0 1 2 3
...（共 16 条 channel）
```

- 4 个 rank 组成环：0→1→2→3→0
- 16 条 channel 并行跑，充分利用 NVLink 带宽
- 适合大数据量 AllReduce（带宽利用率高）
- 实现：reduce-scatter（n-1 步逐块累加） + all-gather（n-1 步广播结果）

### 3.2 Tree AllReduce

```
Trees [0]: Rank 3(根) → Rank 2 → Rank 1 → Rank 0(叶)
```

- 退化为链式结构（4 卡场景，NCCL 认为链式够用）
- 适合小数据量 AllReduce（延迟低，步数少）
- GPU 多时会形成真正的二叉树

### 3.3 NVLS（NVLink SHARP）

```
NVLS multicast support is available on dev 0/1/2/3
NVLS tuning: nChannels 16 chunkSize 131072
```

- 利用 NVSwitch 硬件：GPU 同时发数据到 NVSwitch，硬件直接做 reduce + multicast
- 基本一步完成，不需要多步传递
- H800 + NVSwitch 才能用

### 运行时选择逻辑

```
数据量小（KB 级）  → Tree（低延迟）
数据量中（MB 级）  → Ring（高带宽）
数据量大           → NVLS（硬件加速）
```

每次 `ncclAllReduce` 调用时动态选择，同一次推理中不同层可能走不同算法。

### Channel 分配总结

```
16 coll channels, 16 collnet channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
```

### 传输协议

```
threadThresholds 8/8/64 | 32/8/64 | 512 | 512
```

NCCL 根据数据大小选择传输协议：
- LL（Low Latency）：数据极小，附带校验位
- LL128：128 字节打包，平衡延迟和带宽
- Simple：数据大，直接批量搬，带宽最高

---

## 4. vLLM 层面的通信优化

### 4.1 custom_all_reduce

```
disable_custom_all_reduce=False  → 默认启用
```

vLLM 在 NCCL 之上的自定义 AllReduce，条件自动满足时启用（TP > 1 且 GPU 支持 P2P）。

### 4.2 FlashInfer AllReduce + RMSNorm 融合

```
Enabled custom fusions: allreduce_rms                                    ← 编译时注册融合 pass
Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl ← 运行时初始化
```

链路：
```
vLLM 默认开启 custom_all_reduce
  → 编译时注册 allreduce_rms 融合 pass
  → 运行时初始化 FlashInfer Allreduce workspace（mnnvl 后端）
  → 推理时 AllReduce + RMSNorm 一个 kernel 完成
```

---

## 5. 推理结果

3 条 prompt 全部正确生成：

```
Prompt 0: "What is the capital of France?"
Output:   "The capital of France is Paris."

Prompt 1: "Explain quantum computing in simple terms."
Output:   "Quantum computing is a type of computing that uses the principles of quantum mechanics to process i..."

Prompt 2: "Write a short poem about the ocean."
Output:   "The ocean, vast and blue,..."
```

---

## 6. 完整初始化时间线

```
10:19:05  EngineCore 启动，MultiprocExecutor 并行 spawn 4 个 Worker
10:19:14  Worker 们各自连上 TCP Store（barrier 等待到齐）
10:19:17  NCCL ncclCommInitRank → Bootstrap 信息交换 → 拓扑发现
10:19:17  建立 Ring / Tree / NVLS 拓扑，P2P/CUMEM 通道
10:19:27  FlashInfer AllReduce 初始化
10:19:42  FlashInfer autotune 完成
10:19:45  KV Cache 分配，warmup
10:19:46  torch.compile + CUDAGraph 捕获
10:19:54  引擎就绪
10:19:54  开始推理
```
