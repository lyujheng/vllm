#!/bin/bash
# 实验2: TP=4,PP=1 vs TP=2,PP=2
# 观察: PP 引入的额外延迟，对比纯 TP 方案（都是 4 卡）

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="/home/lvzheng/vllm/learning/week4-distributed/results"
mkdir -p $OUTPUT_DIR

# 避免 localhost 请求走代理
export no_proxy=localhost,127.0.0.1

for CONFIG in "tp4_pp1" "tp2_pp2"; do
    if [ "$CONFIG" == "tp4_pp1" ]; then
        TP=4; PP=1
    else
        TP=2; PP=2
    fi

    echo "============================================"
    echo "Running TP=$TP, PP=$PP"
    echo "============================================"

    vllm serve $MODEL \
        --tensor-parallel-size $TP \
        --pipeline-parallel-size $PP \
        --port 8000 \
        --gpu-memory-utilization 0.85 \
        > $OUTPUT_DIR/${CONFIG}_server.log 2>&1 &
    SERVER_PID=$!

    # 等待服务完全就绪（监听日志中的启动完成标志）
    echo "Waiting for server to start..."
    for i in $(seq 1 120); do
        if grep -q "Application startup complete" $OUTPUT_DIR/${CONFIG}_server.log 2>/dev/null; then
            echo "Server ready!"
            break
        fi
        sleep 5
    done

    if ! grep -q "Application startup complete" $OUTPUT_DIR/${CONFIG}_server.log 2>/dev/null; then
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
        2>&1 | tee $OUTPUT_DIR/${CONFIG}_benchmark.log

    # 关停服务（SIGTERM 正常退出，等待清理）
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
    echo ""
done

echo "Done! Results in $OUTPUT_DIR/"
