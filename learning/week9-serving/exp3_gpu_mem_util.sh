#!/bin/bash
# 实验 3：gpu_memory_utilization 扫描
# 高并发负载，观察 KV cache 容量对 preemption 和吞吐的影响

export CUDA_VISIBLE_DEVICES=2

MODEL="/data/model/Qwen2.5-7B"
PORT=8000
RESULT_DIR="/home/lvzheng/vllm/results/week9/exp3_gpu_mem_util"
mkdir -p "$RESULT_DIR"

GPU_MEM_UTIL_LIST=(0.70 0.80 0.85 0.90 0.95)

for MEM_UTIL in "${GPU_MEM_UTIL_LIST[@]}"; do
    echo "=========================================="
    echo "Testing gpu_memory_utilization=${MEM_UTIL}"
    echo "=========================================="

    # 1) 启动服务
    vllm serve "$MODEL" \
        --port $PORT \
        --gpu-memory-utilization "$MEM_UTIL" \
        --max-num-seqs 64 \
        > "$RESULT_DIR/server_mem${MEM_UTIL}.log" 2>&1 &
    SERVER_PID=$!

    # 等待服务就绪（最多等 600 秒）
    echo "Waiting for server to start..."
    SERVER_READY=0
    for i in $(seq 1 600); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            SERVER_READY=1
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process exited unexpectedly!"
            break
        fi
        sleep 1
    done
    if [ $SERVER_READY -eq 0 ]; then
        echo "Server failed to start, skipping gpu_memory_utilization=${MEM_UTIL}"
        continue
    fi

    # 2) 运行 benchmark（高并发压满）
    vllm bench serve \
        --base-url "http://localhost:${PORT}" \
        --backend openai \
        --endpoint /v1/completions \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 512 \
        --random-output-len 256 \
        --num-prompts 300 \
        --request-rate inf \
        --result-dir "$RESULT_DIR" \
        --result-filename "mem_${MEM_UTIL}.json" \
        2>&1 | tee "$RESULT_DIR/bench_mem${MEM_UTIL}.log"

    # 3) 停止服务
    echo "Stopping server (PID=$SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5

    echo ""
done

echo "实验 3 完成，结果保存在: $RESULT_DIR"
