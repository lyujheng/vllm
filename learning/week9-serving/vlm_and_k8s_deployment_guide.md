# VLM 部署差异与容器化/K8s 部署参考手册

> Week 9 Day 7 交付物。供未来生产部署时快速查阅。

---

## 一、VLM（视觉语言模型）部署差异

### 1.1 与 Dense 模型的核心区别

Dense 文本模型中，每个 token 对应一个 embedding。VLM 中一张图像会被 vision encoder 处理后展开为大量 token：

| 模型 | 单张图像 token 数 | 说明 |
|---|---|---|
| LLaVA-1.5 | 576 | 固定分辨率 |
| LLaVA-NeXT | 1344~2880 | 动态分辨率，取决于图像尺寸 |
| Qwen-VL | 256 | 固定 |
| InternVL | 最高 4096+ | 高分辨率模式 |

这意味着一个带图像的请求，KV cache 消耗可能是纯文本请求的 5~10 倍。

### 1.2 EncoderCacheManager

VLM 有独立于 decoder KV cache 的 **encoder cache**，由 `EncoderCacheManager`（`vllm/v1/core/encoder_cache_manager.py`）管理。

核心设计：
- **独立缓存池**：encoder 输出（如 vision embedding）缓存在专门的池中，与 decoder KV cache 分开管理
- **hash 复用**：相同图像的 encoder 输出通过 mm_hash 识别，多个请求引用同一张图时共享缓存，不重复计算
- **引用计数 + LRU 淘汰**：每个缓存条目有引用计数，引用归零后按 LRU 顺序淘汰
- **粒度**：以单个多模态输入项（如一张图像）为单位管理，而非 token 级别

关键参数：
- `max_num_encoder_input_tokens`（SchedulerConfig）：每步最大 encoder token 预算，控制一次能处理多少图像 token
- encoder cache size：由 `compute_mm_encoder_budget()` 根据模型和 scheduler 配置自动计算

### 1.3 VLM 部署调优要点

```bash
# VLM 部署示例
vllm serve /data/model/llava-v1.5-7b \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90
```

| 参数 | 调整方向 | 原因 |
|---|---|---|
| `max_num_seqs` | **降低**（8~32） | 每个请求 KV cache 消耗大，并发太高容易 OOM |
| `gpu_memory_utilization` | **调高**（0.90~0.95） | 需要更多显存给 KV cache 和 encoder cache |
| `max_num_batched_tokens` | **适当增大** | 图像 token 多，太小会导致单张图都放不进一个 chunk |
| `enable_prefix_caching` | **开启** | system prompt 部分（纯文本）可缓存复用 |
| `limit_mm_per_prompt` | **按需设置** | 限制每个请求最大图像数量，防止单请求吃掉太多资源 |

**容量估算公式**：

```
单请求 KV cache ≈ (text_tokens + image_tokens × num_images) × num_layers × hidden_size × 2(K+V) × dtype_bytes

示例：LLaVA-1.5-7B，1 张图 + 256 text tokens
= (256 + 576) × 32 layers × 4096 × 2 × 2B (FP16)
≈ 1.4 GB KV cache（单请求）
```

### 1.4 VLM vs Dense 部署决策对比

| 维度 | Dense 模型 | VLM |
|---|---|---|
| 单请求 KV cache | 可预测（与 text length 线性） | 波动大（取决于图像数量和分辨率） |
| max_num_seqs | 可设较大（64~256） | 需降低（8~32） |
| 额外缓存 | 无 | EncoderCacheManager |
| Chunked Prefill | chunk 只含 text token | chunk 可能包含大量 image token |
| 预热时间 | 较短 | 较长（vision encoder 额外 CUDA Graph） |

---

## 二、Executor 多后端架构

### 2.1 三种 Executor

EngineCore 通过统一的 `Executor` 接口（`vllm/v1/executor/abstract.py`）调用 Worker，底层实现有三种：

| Executor | 后端标识 | 适用场景 | 进程模型 | 通信方式 |
|---|---|---|---|---|
| `UniProcExecutor` | `"uni"` | TP=1 单卡 | 单进程 | 函数调用 |
| `MultiprocExecutor` | `"mp"` | 单机多卡（TP/PP > 1） | spawn 子进程 | zmq + shared memory + NCCL |
| `RayExecutor` | `"ray"` | 跨节点（多机多卡） | Ray Actor | gRPC + Ray object store + NCCL |

### 2.2 自动选择逻辑

`Executor.get_class()`（`vllm/v1/executor/abstract.py:48`）根据 `distributed_executor_backend` 选择：

```
VllmConfig.__post_init__()
  → 根据 TP/PP size 自动设置 distributed_executor_backend
    → TP=1, PP=1  → "uni"  → UniProcExecutor
    → 单机多卡     → "mp"   → MultiprocExecutor
    → 多节点       → "ray"  → RayExecutor
```

也可通过 `--distributed-executor-backend` 手动指定。

### 2.3 MultiprocExecutor（单机多卡，最常用）

代码路径：`vllm/v1/executor/multiproc_executor.py`

**Worker 生命周期**：
```
spawn 子进程 → init_device(分配 GPU) → load_model(加载权重)
→ ready(通过 pipe 通知父进程) → execute_model loop(持续接收并执行任务)
```

**通信机制**：
- **zmq PUSH/PULL**：传递 SchedulerOutput（调度指令）、控制命令
- **Shared Memory**：传递大 tensor（如 KV cache blocks），避免序列化开销
- **NCCL**：AllReduce/AllGather 等集合通信（TP 切分后的梯度/激活聚合）

**故障检测**：
- **death_pipe**：每个 worker 持有管道的读端，父进程死亡时管道断开，worker 自行清理退出
- **_ensure_worker_termination()**：三级终止（等待 → SIGTERM → SIGKILL）

### 2.4 RayExecutor（跨节点）

代码路径：`vllm/v1/executor/ray_executor.py`

**与 MultiprocExecutor 的差异**：
- Worker 是 Ray Actor，可以分布在不同物理节点上
- 通信经过 Ray 的 gRPC 层和 object store，延迟比本地 zmq 高
- NCCL group 建立需要跨节点协调（通过 Ray 传递 NCCL unique ID）
- 支持 Actor 级别的故障重启

**启动方式**：
```bash
# 节点 1（head node）
ray start --head --port=6379
vllm serve <model> --tensor-parallel-size 16

# 节点 2（worker node）
ray start --address=<head_ip>:6379
# worker 自动加入集群，vLLM 通过 Ray 调度 Actor 到此节点
```

---

## 三、容器化部署

### 3.1 Docker 部署

#### 官方 Dockerfile 结构

vLLM 使用多阶段构建（`docker/Dockerfile`）：

```
base stage         → CUDA + Python 基础环境
csrc-build stage   → 编译 C++/CUDA 扩展（flash-attn, custom kernels）
rust-build stage   → 编译 Rust 前端
build stage        → 构建 Python wheel
vllm-base stage    → 最终运行时镜像（最小依赖）
```

多阶段构建的意义：编译依赖（gcc、cmake、nvcc 等）不进最终镜像，运行时镜像体积小很多。

#### Docker 启动关键参数

```bash
docker run --gpus all \
    --shm-size=16g \
    -v /data/model:/data/model \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    vllm serve /data/model/Qwen2.5-7B \
        --gpu-memory-utilization 0.90
```

| 参数 | 作用 | 注意事项 |
|---|---|---|
| `--gpus all` | 暴露所有 GPU 给容器 | 也可用 `--gpus '"device=0,1"'` 指定 |
| `--shm-size=16g` | 设置 /dev/shm 大小 | **必须设置**。NCCL 多卡通信走 shared memory，默认 64MB 太小会导致通信失败 |
| `-v /data/model:/data/model` | 挂载模型目录 | 避免每次把模型 COPY 进镜像 |
| `--ipc=host` | 共享主机 IPC 命名空间 | 替代 `--shm-size` 的方案，直接使用主机的 /dev/shm |

#### 常见 Docker 问题排查

| 问题 | 现象 | 解决 |
|---|---|---|
| NCCL 通信失败 | `NCCL error: unhandled system error` | 加 `--shm-size=16g` 或 `--ipc=host` |
| GPU 不可见 | `No CUDA GPUs available` | 确认 `--gpus all` 和 nvidia-container-runtime 已安装 |
| 模型加载 OOM | 容器启动后立即被 kill | 检查 `--memory` 限制，模型加载需要大量 CPU 内存 |
| 权限问题 | 无法写入日志/缓存 | 挂载时加 `:rw`，或设置 `--user` |

### 3.2 K8s 部署

#### Health Probe 配置

vLLM 提供 `/health` 端点（`vllm/entrypoints/serve/instrumentator/health.py`），内部调用 `client.check_health()` 检查引擎状态：
- 引擎正常 → 返回 200
- 引擎异常（EngineDeadError）→ 返回 503

DP Supervisor 模式下，`/health` 会聚合所有 DP worker 的健康状态（`vllm/entrypoints/openai/dp_supervisor.py:225`）。

```yaml
# K8s Pod spec 示例
containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    ports:
      - containerPort: 8000
    resources:
      limits:
        nvidia.com/gpu: "1"    # 声明 GPU 资源
        memory: "32Gi"         # CPU 内存，模型加载需要
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 300  # 模型加载 + warmup 需要时间
      periodSeconds: 10
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 300
      periodSeconds: 5
      failureThreshold: 1       # 一次失败就摘流量
```

**Liveness vs Readiness**：
- **Liveness**：进程是否存活。失败会重启 Pod。`initialDelaySeconds` 要足够大，覆盖模型加载时间
- **Readiness**：是否可以接收流量。失败会从 Service endpoints 中摘除，但不重启。模型加载完成前 readiness 会失败，流量不会打到未就绪的 Pod

#### Graceful Shutdown 流程

K8s 与 vLLM 的关闭配合链路（`vllm/entrypoints/launcher.py:107-146`）：

```
K8s 发送 SIGTERM
  → launcher.py signal_handler() 捕获
    → shutdown_event.set()
      → handle_shutdown() 被唤醒
        → engine_client.shutdown(timeout=shutdown_timeout)
          → 如果 timeout > 0: wait_for_requests_to_drain() 等待请求排空
          → 如果 timeout = 0: 立即 abort 所有请求
        → server.should_exit = True
          → HTTP server 停止
```

**关键配置**：

```yaml
# K8s Pod spec
spec:
  terminationGracePeriodSeconds: 600  # K8s 等待时间（默认 30s 太短）
```

```bash
# vLLM 启动参数
vllm serve <model> --shutdown-timeout 300
```

**必须满足**：`terminationGracePeriodSeconds` > `--shutdown-timeout`

否则 vLLM 还在等请求排空，K8s 就发 SIGKILL 强制杀掉，导致在途请求中断。

| 参数 | 默认值 | 建议值 | 说明 |
|---|---|---|---|
| `--shutdown-timeout` | 0（立即 abort） | 300 | 等待在途请求完成的秒数 |
| `terminationGracePeriodSeconds` | 30 | 600 | K8s 等待 Pod 终止的秒数 |

#### 滚动更新（Rolling Update）

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 新 Pod 先启动
      maxUnavailable: 0  # 旧 Pod 只在新 Pod ready 后才下线
```

流程：
1. 新 Pod 创建 → 加载模型 → warmup → readiness probe 通过 → 加入 Service endpoints
2. 旧 Pod 收到 SIGTERM → 从 endpoints 摘除 → drain 在途请求 → shutdown
3. 全程对外无感知中断

注意：LLM 服务启动慢（模型加载 + CUDA Graph warmup），`maxSurge=1` 意味着需要额外一张卡/节点的资源来做滚动更新。

#### 关键 K8s 配置清单

| 配置项 | 说明 |
|---|---|
| `nvidia.com/gpu: N` | GPU 资源声明，N = TP × PP |
| `memory` / `ephemeral-storage` | CPU 内存和临时存储，模型加载和 tokenizer 需要 |
| `nodeSelector` / `tolerations` | 确保调度到 GPU 节点 |
| `volumes` + `volumeMounts` | 挂载模型权重（HostPath / PVC / NFS） |
| `HPA` | 基于自定义指标（QPS、GPU 利用率、`num_requests_waiting`）的自动扩缩容 |
| `PodDisruptionBudget` | 保证滚动更新/节点维护时最少可用 Pod 数 |

---

## 四、部署决策快速参考

```
模型类型？
├─ Dense text → 标准配置
└─ VLM → 降 max_num_seqs、升 gpu_memory_utilization、注意 encoder cache

部署规模？
├─ 单卡 → UniProcExecutor，最简单
├─ 单机多卡 → MultiprocExecutor（自动选择）
└─ 多节点 → RayExecutor，需要 Ray 集群

运行环境？
├─ 裸机 → 直接 vllm serve
├─ Docker → 注意 --shm-size 和 --gpus
└─ K8s → 配置 health probe + graceful shutdown + 滚动更新
```
