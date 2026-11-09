#!/bin/bash
# 实验 1：max_num_seqs 扫描
# 固定模型 Qwen2.5-7B (TP=1)，扫描 max_num_seqs 对吞吐和延迟的影响
# 使用 random dataset，固定 input/output 长度，request_rate=inf（压满）

export CUDA_VISIBLE_DEVICES=2

MODEL="/data/model/Qwen2.5-7B"
PORT=8000
RESULT_DIR="/home/lvzheng/vllm/results/week9/exp1_max_num_seqs"
mkdir -p "$RESULT_DIR"

MAX_NUM_SEQS_LIST=(4 8 16 32 64 128)

for MAX_SEQS in "${MAX_NUM_SEQS_LIST[@]}"; do
    echo "=========================================="
    echo "Testing max_num_seqs=${MAX_SEQS}"
    echo "=========================================="

    # 1) 启动服务
    vllm serve "$MODEL" \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$MAX_SEQS" \
        > "$RESULT_DIR/server_seqs${MAX_SEQS}.log" 2>&1 &
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
        # 检查服务进程是否已退出
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process exited unexpectedly!"
            break
        fi
        sleep 1
    done
    if [ $SERVER_READY -eq 0 ]; then
        echo "Server failed to start, skipping max_num_seqs=${MAX_SEQS}"
        continue
    fi

    # 2) 运行 benchmark
    vllm bench serve \
        --base-url "http://localhost:${PORT}" \
        --backend openai \
        --endpoint /v1/completions \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 512 \
        --random-output-len 128 \
        --num-prompts 200 \
        --request-rate inf \
        --result-dir "$RESULT_DIR" \
        --result-filename "seqs_${MAX_SEQS}.json" \
        2>&1 | tee "$RESULT_DIR/bench_seqs${MAX_SEQS}.log"

    # 3) 停止服务
    echo "Stopping server (PID=$SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5

    echo ""
done

echo "实验 1 完成，结果保存在: $RESULT_DIR"
