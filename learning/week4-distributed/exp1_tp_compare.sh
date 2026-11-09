#!/bin/bash
# 实验1: TP=1 vs TP=2 vs TP=4 vs TP=8 推理对比
# 观察: throughput, latency, 通信开销随 TP 增大的变化

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="/home/lvzheng/vllm/learning/week4-distributed/results"
mkdir -p $OUTPUT_DIR

# 避免 localhost 请求走代理
export no_proxy=localhost,127.0.0.1

for TP in 1 2 4; do
    echo "============================================"
    echo "Running TP=$TP"
    echo "============================================"

    vllm serve $MODEL \
        --tensor-parallel-size $TP \
        --port 8000 \
        --gpu-memory-utilization 0.85 \
        > $OUTPUT_DIR/tp${TP}_server.log 2>&1 &
    SERVER_PID=$!

    # 等待服务完全就绪（监听日志中的启动完成标志）
    echo "Waiting for server to start..."
    for i in $(seq 1 120); do
        if grep -q "Application startup complete" $OUTPUT_DIR/tp${TP}_server.log 2>/dev/null; then
            echo "Server ready!"
            break
        fi
        sleep 5
    done

    if ! grep -q "Application startup complete" $OUTPUT_DIR/tp${TP}_server.log 2>/dev/null; then
        echo "ERROR: Server failed to start after 600 seconds"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi

    # 用 vLLM benchmark 工具跑
    vllm bench serve \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts 100 \
        --request-rate inf \
        --port 8000 \
        2>&1 | tee $OUTPUT_DIR/tp${TP}_benchmark.log

    # 关停服务（SIGTERM 正常退出，等待清理）
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
    echo ""
done

echo "Done! Results in $OUTPUT_DIR/"
