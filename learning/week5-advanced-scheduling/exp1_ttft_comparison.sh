#!/bin/bash
# Exp1: Chunked Prefill 对 TTFT 和 ITL 的影响
# 对比不同 max_num_batched_tokens 配置下的延迟指标
#
# 核心思路：用长 prompt 并发请求，观察：
# - TTFT: chunked prefill 如何影响首 token 延迟
# - ITL:  prefill 和 decode 混合 batch 时，decode 请求的 token 间延迟

set -e

# 指定使用 GPU 7
export CUDA_VISIBLE_DEVICES=7

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="/home/lvzheng/vllm/learning/week5-advanced-scheduling/results"
mkdir -p $OUTPUT_DIR

export no_proxy=localhost,127.0.0.1

run_test() {
    local config_name=$1
    local max_tokens=$2
    local enable_chunked=$3
    local port=8000

    echo ""
    echo "============================================"
    echo "Config: $config_name"
    echo "  max_num_batched_tokens=$max_tokens"
    echo "  enable_chunked_prefill=$enable_chunked"
    echo "============================================"

    # 启动 vLLM server
    local extra_args=""
    if [ "$enable_chunked" = "false" ]; then
        extra_args="--no-enable-chunked-prefill"
    fi

    vllm serve $MODEL \
        --port $port \
        --max-model-len 8192 \
        --max-num-batched-tokens $max_tokens \
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

    # 用长 prompt + 并发请求，突出 chunked prefill 效果
    # random-input-len=2048: 让 prefill 足够长
    # random-output-len=128: 让 decode 持续一段时间
    # num-prompts=30, request-rate=5: 并发产生混合 batch
    vllm bench serve \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 2048 \
        --random-output-len 128 \
        --num-prompts 30 \
        --request-rate 5 \
        --port $port \
        2>&1 | tee $OUTPUT_DIR/${config_name}_benchmark.log

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
}

# 实验1: 不开 chunked prefill (baseline)
run_test "no_chunked" 8192 "false"

# 实验2: chunked prefill, 大 budget (chunk 较大)
run_test "chunked_8192" 8192 "true"

# 实验3: chunked prefill, 中等 budget
run_test "chunked_2048" 2048 "true"

# 实验4: chunked prefill, 小 budget (chunk 较小)
run_test "chunked_1024" 1024 "true"

echo ""
echo "============================================"
echo "All tests done! Results in $OUTPUT_DIR/"
echo ""
echo "Compare TTFT and ITL across configs:"
for f in $OUTPUT_DIR/*_benchmark.log; do
    echo ""
    echo "--- $(basename $f) ---"
    grep -E "TTFT|ITL|Throughput" $f || true
done
