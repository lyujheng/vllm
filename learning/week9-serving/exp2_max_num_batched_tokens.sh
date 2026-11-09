#!/bin/bash
# 实验 2：max_num_batched_tokens 扫描（开启 chunked prefill）
# 使用混合长度负载，观察 TTFT 和 ITL 的权衡

export CUDA_VISIBLE_DEVICES=2

MODEL="/data/model/Qwen2.5-7B"
PORT=8000
RESULT_DIR="/home/lvzheng/vllm/results/week9/exp2_max_num_batched_tokens"
mkdir -p "$RESULT_DIR"

BATCHED_TOKENS_LIST=(512 1024 2048 4096 8192)

for TOKENS in "${BATCHED_TOKENS_LIST[@]}"; do
    echo "=========================================="
    echo "Testing max_num_batched_tokens=${TOKENS}"
    echo "=========================================="

    # 1) 启动服务（chunked prefill 默认开启）
    vllm serve "$MODEL" \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 64 \
        --max-num-batched-tokens "$TOKENS" \
        > "$RESULT_DIR/server_tokens${TOKENS}.log" 2>&1 &
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
        echo "Server failed to start, skipping max_num_batched_tokens=${TOKENS}"
        continue
    fi

    # 2) 运行 benchmark（混合长度：input 256~2048）
    vllm bench serve \
        --base-url "http://localhost:${PORT}" \
        --backend openai \
        --endpoint /v1/completions \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 128 \
        --num-prompts 200 \
        --request-rate 10 \
        --result-dir "$RESULT_DIR" \
        --result-filename "tokens_${TOKENS}.json" \
        2>&1 | tee "$RESULT_DIR/bench_tokens${TOKENS}.log"

    # 3) 停止服务
    echo "Stopping server (PID=$SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5

    echo ""
done

echo "实验 2 完成，结果保存在: $RESULT_DIR"
