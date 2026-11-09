#!/bin/bash
# Week 5 Day 7 实验4：组合优化效果对比
# 对比不同优化组合的性能

set -e

RESULTS_DIR="/home/lvzheng/vllm/learning/week5-advanced-scheduling/results"
mkdir -p $RESULTS_DIR

MODEL="/data/model/Qwen2.5-7B"

echo "=========================================="
echo "实验4：组合优化效果对比"
echo "=========================================="

# 清除旧的 server 日志，避免误匹配
rm -f $RESULTS_DIR/exp4_*_server.log

# 测试用例：包含相同 prefix 的多个请求
TEST_SCRIPT='
import time
import requests
import json

url = "http://localhost:8000/v1/completions"

# 3个请求有相同的 system prompt（测试 prefix caching）
prompts = [
    "你是一个专业的AI助手。请回答：什么是机器学习？",
    "你是一个专业的AI助手。请回答：什么是深度学习？",
    "你是一个专业的AI助手。请回答：什么是强化学习？"
]

results = []
overall_start = time.time()

for i, prompt in enumerate(prompts):
    data = {
        "model": "/data/model/Qwen2.5-7B",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.0
    }

    start = time.time()
    resp = requests.post(url, json=data, timeout=60)
    latency = time.time() - start

    result = resp.json()
    tokens = len(result["choices"][0]["text"].split())

    results.append({
        "prompt_id": i,
        "latency": latency,
        "tokens": tokens,
        "ttft_ms": latency * 1000  # 简化假设
    })
    print(f"Request {i}: latency={latency:.2f}s, tokens={tokens}")

overall_time = time.time() - overall_start

with open(output_file, "w") as f:
    json.dump({
        "results": results,
        "overall_time": overall_time,
        "throughput_tokens_per_sec": sum(r["tokens"] for r in results) / overall_time
    }, f, indent=2)

avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
throughput = sum(r["tokens"] for r in results) / overall_time
print(f"\nAverage TTFT: {avg_ttft:.2f} ms")
print(f"Throughput: {throughput:.2f} tokens/s")
'

# 配置1: 所有优化关闭
echo ""
echo "[1/4] Baseline: 所有优化关闭"
vllm serve $MODEL \
    --enforce-eager \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    > $RESULTS_DIR/exp4_baseline_server.log 2>&1 &

SERVER_PID=$!

# 等待服务完全就绪（监听日志中的启动完成标志）
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $RESULTS_DIR/exp4_baseline_server.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

if ! grep -q "Application startup complete" $RESULTS_DIR/exp4_baseline_server.log 2>/dev/null; then
    echo "ERROR: Server failed to start after 600 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

python3 << EOF
output_file = "./results/exp4_baseline.json"
$TEST_SCRIPT
EOF

# 关停服务（SIGTERM 正常退出，等待清理）
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

# 配置2: 只开启 CUDA Graph
echo ""
echo "[2/4] 只开启 CUDA Graph"
vllm serve $MODEL \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    > $RESULTS_DIR/exp4_cudagraph_only_server.log 2>&1 &

SERVER_PID=$!

# 等待服务完全就绪（监听日志中的启动完成标志）
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $RESULTS_DIR/exp4_cudagraph_only_server.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

if ! grep -q "Application startup complete" $RESULTS_DIR/exp4_cudagraph_only_server.log 2>/dev/null; then
    echo "ERROR: Server failed to start after 600 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

python3 << EOF
output_file = "./results/exp4_cudagraph_only.json"
$TEST_SCRIPT
EOF

# 关停服务（SIGTERM 正常退出，等待清理）
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

# 配置3: CUDA Graph + Prefix Caching
echo ""
echo "[3/4] CUDA Graph + Prefix Caching"
vllm serve $MODEL \
    --enable-prefix-caching \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    > $RESULTS_DIR/exp4_cudagraph_prefix_server.log 2>&1 &

SERVER_PID=$!

# 等待服务完全就绪（监听日志中的启动完成标志）
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $RESULTS_DIR/exp4_cudagraph_prefix_server.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

if ! grep -q "Application startup complete" $RESULTS_DIR/exp4_cudagraph_prefix_server.log 2>/dev/null; then
    echo "ERROR: Server failed to start after 600 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

python3 << EOF
output_file = "./results/exp4_cudagraph_prefix.json"
$TEST_SCRIPT
EOF

# 关停服务（SIGTERM 正常退出，等待清理）
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

# 配置4: 所有优化开启 (CUDA Graph + Prefix Caching + Chunked Prefill)
echo ""
echo "[4/4] 所有优化开启"
vllm serve $MODEL \
    --enable-prefix-caching \
    --max-num-batched-tokens 512 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    > $RESULTS_DIR/exp4_all_optimizations_server.log 2>&1 &

SERVER_PID=$!

# 等待服务完全就绪（监听日志中的启动完成标志）
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $RESULTS_DIR/exp4_all_optimizations_server.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

if ! grep -q "Application startup complete" $RESULTS_DIR/exp4_all_optimizations_server.log 2>/dev/null; then
    echo "ERROR: Server failed to start after 600 seconds"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

python3 << EOF
output_file = "./results/exp4_all_optimizations.json"
$TEST_SCRIPT
EOF

# 关停服务（SIGTERM 正常退出，等待清理）
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

# 生成对比报告
echo ""
echo "[5/4] 生成对比报告"
python3 << 'EOF'
import json

configs = [
    ("baseline", "Baseline (所有优化关闭)"),
    ("cudagraph_only", "CUDA Graph Only"),
    ("cudagraph_prefix", "CUDA Graph + Prefix Caching"),
    ("all_optimizations", "所有优化开启")
]

print("=" * 80)
print("组合优化性能对比")
print("=" * 80)
print(f"{'配置':<30} {'平均TTFT(ms)':<15} {'吞吐(tok/s)':<15} {'总耗时(s)':<10}")
print("-" * 80)

baseline_throughput = None
results_summary = []

for key, name in configs:
    try:
        with open(f"./results/exp4_{key}.json") as f:
            data = json.load(f)

        avg_ttft = sum(r["ttft_ms"] for r in data["results"]) / len(data["results"])
        throughput = data["throughput_tokens_per_sec"]
        overall_time = data["overall_time"]

        if baseline_throughput is None:
            baseline_throughput = throughput

        speedup = throughput / baseline_throughput if baseline_throughput > 0 else 1.0

        print(f"{name:<30} {avg_ttft:<15.2f} {throughput:<15.2f} {overall_time:<10.2f}")

        results_summary.append({
            "config": name,
            "avg_ttft_ms": avg_ttft,
            "throughput": throughput,
            "speedup": speedup
        })
    except Exception as e:
        print(f"{name:<30} Error: {e}")

print("=" * 80)

# 保存汇总
with open("./results/exp4_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n汇总报告已保存到 ./results/exp4_summary.json")
EOF

echo ""
echo "实验完成！结果保存在 ./results/exp4_*.json"
