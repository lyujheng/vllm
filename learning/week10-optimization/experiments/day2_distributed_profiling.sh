#!/bin/bash
# Day 2: 分布式 Profiling — TP=2 vs TP=1 对比实验
# 目标：观察 TP=2 场景下 NCCL 通信开销，对比 TP=1 的 kernel 分布

set +e

MODEL="/data/model/Qwen2.5-7B"
BASE_DIR="/home/lvzheng/vllm/learning/week10-optimization"
RESULTS_DIR="${BASE_DIR}/results/day2_traces"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${RESULTS_DIR}/tp1" "${RESULTS_DIR}/tp2" "${LOG_DIR}"

PORT_TP1=8201
PORT_TP2=8202

# =============================================
# 实验1: TP=1 基线 profiling
# =============================================
echo "========================================="
echo "Step 1: 启动 TP=1 server"
echo "========================================="

vllm serve "${MODEL}" \
    --port ${PORT_TP1} \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "${RESULTS_DIR}/tp1" \
    --profiler-config.torch_profiler_with_flops true \
    --profiler-config.torch_profiler_use_gzip true \
    --profiler-config.torch_profiler_dump_cuda_time_total true \
    > "${LOG_DIR}/day2_tp1_server.log" 2>&1 &
TP1_PID=$!
echo "TP=1 server PID: ${TP1_PID}"

echo "等待 TP=1 server 启动..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT_TP1}/health > /dev/null 2>&1; then
        echo "TP=1 server 就绪 (${i}s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: TP=1 server 启动超时"
        kill ${TP1_PID} 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo ""
echo "========================================="
echo "Step 2: TP=1 profiling — prefill 场景"
echo "========================================="

# 启动 profiler
curl -s -X POST http://localhost:${PORT_TP1}/start_profile
echo " -> profiler started"
sleep 1

# 发送 prefill-heavy 请求 (长 prompt, 短 output)
LONG_PROMPT="Please analyze the following passage in great detail and provide a comprehensive summary. The history of computing dates back to ancient civilizations that developed tools for mathematical calculations. The abacus, developed around 2400 BC, was one of the earliest computing devices. In the 17th century, mechanical calculators were invented by Blaise Pascal and Gottfried Wilhelm Leibniz. Charles Babbage designed the Analytical Engine in the 1830s, which contained many features of modern computers. Ada Lovelace wrote the first algorithm intended for machine processing. The development of electronic computers began in the 1940s with machines like ENIAC and Colossus."

curl -s http://localhost:${PORT_TP1}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"${LONG_PROMPT}\",
        \"max_tokens\": 16,
        \"temperature\": 0
    }" > "${RESULTS_DIR}/tp1_prefill_response.json"
echo " -> prefill request done"

sleep 2

# 停止 profiler
curl -s -X POST http://localhost:${PORT_TP1}/stop_profile
echo " -> profiler stopped"
sleep 3

echo ""
echo "========================================="
echo "Step 3: 关闭 TP=1 server"
echo "========================================="
kill ${TP1_PID} 2>/dev/null
wait ${TP1_PID} 2>/dev/null
echo "TP=1 server 已关闭"
sleep 3

# =============================================
# 实验2: TP=2 profiling
# =============================================
echo ""
echo "========================================="
echo "Step 4: 启动 TP=2 server"
echo "========================================="

vllm serve "${MODEL}" \
    --port ${PORT_TP2} \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "${RESULTS_DIR}/tp2" \
    --profiler-config.torch_profiler_with_flops true \
    --profiler-config.torch_profiler_use_gzip true \
    --profiler-config.torch_profiler_dump_cuda_time_total true \
    > "${LOG_DIR}/day2_tp2_server.log" 2>&1 &
TP2_PID=$!
echo "TP=2 server PID: ${TP2_PID}"

echo "等待 TP=2 server 启动..."
for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT_TP2}/health > /dev/null 2>&1; then
        echo "TP=2 server 就绪 (${i}s)"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: TP=2 server 启动超时"
        kill ${TP2_PID} 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo ""
echo "========================================="
echo "Step 5: TP=2 profiling — 同样的 prefill 场景"
echo "========================================="

curl -s -X POST http://localhost:${PORT_TP2}/start_profile
echo " -> profiler started"
sleep 1

# 发送同样的请求
curl -s http://localhost:${PORT_TP2}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"${LONG_PROMPT}\",
        \"max_tokens\": 16,
        \"temperature\": 0
    }" > "${RESULTS_DIR}/tp2_prefill_response.json"
echo " -> prefill request done"

sleep 2

curl -s -X POST http://localhost:${PORT_TP2}/stop_profile
echo " -> profiler stopped"
sleep 3

echo ""
echo "========================================="
echo "Step 6: 关闭 TP=2 server"
echo "========================================="
kill ${TP2_PID} 2>/dev/null
wait ${TP2_PID} 2>/dev/null
echo "TP=2 server 已关闭"

echo ""
echo "========================================="
echo "完成！Trace 文件位于:"
echo "  TP=1: ${RESULTS_DIR}/tp1/"
echo "  TP=2: ${RESULTS_DIR}/tp2/"
echo "========================================="
ls -lh "${RESULTS_DIR}/tp1/" 2>/dev/null
ls -lh "${RESULTS_DIR}/tp2/" 2>/dev/null
