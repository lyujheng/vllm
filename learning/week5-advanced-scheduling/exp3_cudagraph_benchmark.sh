#!/bin/bash
# Week 5 Day 7 实验3：CUDA Graph 性能对比
# 对比开启/关闭 CUDA Graph 对 decode 性能的影响

set -e

RESULTS_DIR="/home/lvzheng/vllm/learning/week5-advanced-scheduling/results"
mkdir -p $RESULTS_DIR

MODEL="/data/model/Qwen2.5-7B"
PROMPT="请详细介绍一下人工智能的发展历史和未来趋势。"

echo "=========================================="
echo "实验3：CUDA Graph 性能对比"
echo "=========================================="

# 1. Baseline: 关闭 CUDA Graph
echo ""
echo "[1/3] Baseline: 关闭 CUDA Graph (enforce_eager=True)"
vllm serve $MODEL \
    --enforce-eager \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    2>&1 | tee $RESULTS_DIR/exp3_eager_server.log &

# 获取 vllm serve 的真实 PID（不是 tee 的 PID）
sleep 2
SERVER_PID=$(pgrep -f "vllm serve.*$MODEL.*--enforce-eager")
echo "Server PID: $SERVER_PID"
echo "Waiting for server to start..."

# 监听日志中的启动完成标志
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $RESULTS_DIR/exp3_eager_server.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

if ! grep -q "Application startup complete" $RESULTS_DIR/exp3_eager_server.log 2>/dev/null; then
    echo "ERROR: Server failed to start after 600 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

sleep 5  # 额外等待确保完全就绪

# 发送请求并测量
python3 << 'EOF'
import time
import requests
import json

url = "http://localhost:8000/v1/completions"
data = {
    "model": "/data/model/Qwen2.5-7B",
    "prompt": "请详细介绍一下人工智能的发展历史和未来趋势。",
    "max_tokens": 200,
    "temperature": 0.0,
    "n": 1
}

results = []
for i in range(10):
    start = time.time()
    resp = requests.post(url, json=data, timeout=60)
    latency = time.time() - start

    result = resp.json()

    # 检查响应格式
    if "choices" not in result:
        print(f"Error response: {result}")
        continue

    tokens = len(result["choices"][0]["text"].split())
    tpot = latency / tokens if tokens > 0 else 0

    results.append({
        "iteration": i,
        "latency": latency,
        "tokens": tokens,
        "tpot_ms": tpot * 1000
    })
    print(f"Iter {i}: {latency:.2f}s, {tokens} tokens, TPOT: {tpot*1000:.2f}ms")

with open("./results/exp3_eager.json", "w") as f:
    json.dump(results, f, indent=2)

avg_tpot = sum(r["tpot_ms"] for r in results) / len(results)
print(f"\nAverage TPOT: {avg_tpot:.2f} ms/token")
EOF

# 关停服务（SIGTERM 正常退出，等待清理）
echo "Stopping baseline server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

# 2. 开启 CUDA Graph（默认）
echo ""
echo "[2/3] 开启 CUDA Graph (默认配置)"
vllm serve $MODEL \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    2>&1 | tee $RESULTS_DIR/exp3_cudagraph_server.log &

# 获取 vllm serve 的真实 PID（不是 tee 的 PID）
sleep 2
SERVER_PID=$(pgrep -f "vllm serve.*$MODEL" | grep -v "enforce-eager")
echo "Server PID: $SERVER_PID"
echo "Waiting for server to start..."

# 监听日志中的启动完成标志
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $RESULTS_DIR/exp3_cudagraph_server.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

if ! grep -q "Application startup complete" $RESULTS_DIR/exp3_cudagraph_server.log 2>/dev/null; then
    echo "ERROR: Server failed to start after 600 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

sleep 5  # 额外等待确保完全就绪

python3 << 'EOF'
import time
import requests
import json

url = "http://localhost:8000/v1/completions"
data = {
    "model": "/data/model/Qwen2.5-7B",
    "prompt": "请详细介绍一下人工智能的发展历史和未来趋势。",
    "max_tokens": 200,
    "temperature": 0.0,
    "n": 1
}

results = []
for i in range(10):
    start = time.time()
    resp = requests.post(url, json=data, timeout=60)
    latency = time.time() - start

    result = resp.json()

    # 检查响应格式
    if "choices" not in result:
        print(f"Error response: {result}")
        continue

    tokens = len(result["choices"][0]["text"].split())
    tpot = latency / tokens if tokens > 0 else 0

    results.append({
        "iteration": i,
        "latency": latency,
        "tokens": tokens,
        "tpot_ms": tpot * 1000
    })
    print(f"Iter {i}: {latency:.2f}s, {tokens} tokens, TPOT: {tpot*1000:.2f}ms")

with open("./results/exp3_cudagraph.json", "w") as f:
    json.dump(results, f, indent=2)

avg_tpot = sum(r["tpot_ms"] for r in results) / len(results)
print(f"\nAverage TPOT: {avg_tpot:.2f} ms/token")
EOF

kill $SERVER_PID
sleep 5

# 3. 对比分析
echo ""
echo "[3/3] 对比分析"
python3 << 'EOF'
import json

with open("./results/exp3_eager.json") as f:
    eager = json.load(f)
with open("./results/exp3_cudagraph.json") as f:
    cudagraph = json.load(f)

eager_tpot = sum(r["tpot_ms"] for r in eager) / len(eager)
cuda_tpot = sum(r["tpot_ms"] for r in cudagraph) / len(cudagraph)

speedup = eager_tpot / cuda_tpot

print("=" * 60)
print("CUDA Graph 性能对比")
print("=" * 60)
print(f"Eager 模式 TPOT:      {eager_tpot:.2f} ms/token")
print(f"CUDA Graph TPOT:     {cuda_tpot:.2f} ms/token")
print(f"加速比:              {speedup:.2f}x")
print("=" * 60)

with open("./results/exp3_summary.txt", "w") as f:
    f.write("CUDA Graph Performance Comparison\n")
    f.write("=" * 60 + "\n")
    f.write(f"Eager Mode TPOT:     {eager_tpot:.2f} ms/token\n")
    f.write(f"CUDA Graph TPOT:     {cuda_tpot:.2f} ms/token\n")
    f.write(f"Speedup:             {speedup:.2f}x\n")
EOF

echo ""
echo "实验完成！结果保存在 ./results/exp3_*.json"
