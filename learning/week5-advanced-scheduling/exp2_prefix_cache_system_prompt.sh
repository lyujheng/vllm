#!/bin/bash
# Exp2: Prefix Caching 对共享前缀的影响
# 修复：使用 --random-prefix-len 确保请求之间有共享前缀

set -e

export CUDA_VISIBLE_DEVICES=7

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="/home/lvzheng/vllm/learning/week5-advanced-scheduling/results"
mkdir -p $OUTPUT_DIR

export no_proxy=localhost,127.0.0.1

run_test() {
    local config_name=$1
    local enable_prefix_cache=$2
    local port=8000

    echo ""
    echo "============================================"
    echo "Config: $config_name"
    echo "  enable_prefix_caching=$enable_prefix_cache"
    echo "============================================"

    local extra_args=""
    if [ "$enable_prefix_cache" = "false" ]; then
        extra_args="--no-enable-prefix-caching"
    fi

    vllm serve $MODEL \
        --port $port \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --disable-uvicorn-access-log \
        $extra_args \
        > $OUTPUT_DIR/${config_name}_server.log 2>&1 &
    SERVER_PID=$!

    echo "Waiting for server to start (PID: $SERVER_PID)..."
    for i in $(seq 1 120); do
        if grep -q "Application startup complete" $OUTPUT_DIR/${config_name}_server.log 2>/dev/null; then
            echo "Server ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "ERROR: Server failed to start"
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi
        sleep 5
    done

    # 关键修复：--random-prefix-len=1024
    # 所有 20 个请求共享前 1024 个 token，后 512 个 token 随机不同
    # 这样 prefix cache 才有命中机会
    vllm bench serve \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 1536 \
        --random-output-len 128 \
        --random-prefix-len 1024 \
        --num-prompts 20 \
        --request-rate 2 \
        --port $port \
        2>&1 | tee $OUTPUT_DIR/${config_name}_benchmark.log

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
}

# 实验1: 开启 prefix caching（应该能命中共享的 1024 token 前缀）
run_test "prefix_cache_on_v2" "true"

# 实验2: 关闭 prefix caching（每次都重新计算完整 1536 token）
run_test "prefix_cache_off_v2" "false"

echo ""
echo "============================================"
echo "Results comparison:"
echo ""
echo "--- Prefix Cache ON ---"
grep -E "TTFT|Throughput" $OUTPUT_DIR/prefix_cache_on_v2_benchmark.log || true
echo ""
echo "--- Prefix Cache OFF ---"
grep -E "TTFT|Throughput" $OUTPUT_DIR/prefix_cache_off_v2_benchmark.log || true
