#!/bin/bash
# Day 1: Qwen2.5-7B TP=1 torch profiler 实验
# 目标：观察 Prefill vs Decode 的 kernel 分布和各组件耗时占比
#
# 实验流程：
#   1. 启动带 torch profiler 的 vLLM server
#   2. 等待服务就绪
#   3. 发送请求触发推理
#   4. 通过 API 控制 profiling 采集窗口
#   5. 停止服务，查看 trace 文件
#
# 使用方法：
#   bash experiments/day1_profiling.sh

set +e  # 不因单个命令失败退出，后台任务可能返回非零码

export no_proxy=localhost,127.0.0.1

# ============================================
# 配置
# ============================================
MODEL="/data/model/Qwen2.5-7B"
PORT=8000
PROFILER_DIR="/home/lvzheng/vllm/learning/week10-optimization/results/day1_traces"
LOG_DIR="/home/lvzheng/vllm/learning/week10-optimization/logs"
LOG_FILE="${LOG_DIR}/day1_server.log"

mkdir -p "${PROFILER_DIR}" "${LOG_DIR}"

# ============================================
# Step 1: 启动带 torch profiler 的 vLLM server
# ============================================
echo "=========================================="
echo "Step 1: 启动 vLLM server (TP=1, torch profiler)"
echo "=========================================="

vllm serve "${MODEL}" \
    --port ${PORT} \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "${PROFILER_DIR}" \
    --profiler-config.torch_profiler_with_stack true \
    --profiler-config.torch_profiler_with_flops true \
    --profiler-config.torch_profiler_record_shapes true \
    --profiler-config.delay_iterations 2 \
    --profiler-config.warmup_iterations 1 \
    --profiler-config.active_iterations 3 \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ============================================
# Step 2: 等待服务就绪
# ============================================
echo ""
echo "=========================================="
echo "Step 2: 等待服务就绪..."
echo "=========================================="

for i in $(seq 1 120); do
    if grep -q "Application startup complete" "${LOG_FILE}" 2>/dev/null; then
        echo "✓ Server ready! (${i}*5s)"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "✗ Server process died. Check ${LOG_FILE}"
        exit 1
    fi
    sleep 5
done

if ! grep -q "Application startup complete" "${LOG_FILE}" 2>/dev/null; then
    echo "✗ Timeout waiting for server. Check ${LOG_FILE}"
    kill ${SERVER_PID} 2>/dev/null
    exit 1
fi

# ============================================
# Step 3: 开始 profiling
# ============================================
echo ""
echo "=========================================="
echo "Step 3: 开始 profiling"
echo "=========================================="

curl -s -X POST http://localhost:${PORT}/start_profile
echo ""
echo "✓ Profiling started (delay=2 iters, warmup=1 iter, active=3 iters)"

# ============================================
# Step 4: 发送 Prefill 密集型请求（长 prompt，短输出）
# ============================================
echo ""
echo "=========================================="
echo "Step 4: 发送 Prefill 密集型请求"
echo "=========================================="

# 生成约 1024 token 的长 prompt
LONG_PROMPT="Please analyze the following text in detail and provide a comprehensive summary. "
for i in $(seq 1 60); do
    LONG_PROMPT="${LONG_PROMPT}The quick brown fox jumps over the lazy dog. This is sentence number ${i} for padding. "
done

echo "Sending prefill-heavy request (long prompt, max_tokens=16)..."
curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"${LONG_PROMPT}\",
        \"max_tokens\": 16,
        \"temperature\": 0.0
    }" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || true
echo ""

# ============================================
# Step 5: 发送 Decode 密集型请求（短 prompt，长输出）
# ============================================
echo ""
echo "=========================================="
echo "Step 5: 发送 Decode 密集型请求"
echo "=========================================="

echo "Sending decode-heavy request (short prompt, max_tokens=512)..."
curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"Write a detailed essay about the history of artificial intelligence.\",
        \"max_tokens\": 512,
        \"temperature\": 0.0
    }" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || true
echo ""

# ============================================
# Step 6: 发送并发请求（模拟 batching）
# ============================================
echo ""
echo "=========================================="
echo "Step 6: 发送并发请求（模拟 continuous batching）"
echo "=========================================="

echo "Sending 4 concurrent requests..."
for i in $(seq 1 4); do
    curl -s http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"Tell me about topic number ${i} in great detail.\",
            \"max_tokens\": 128,
            \"temperature\": 0.0
        }" > /dev/null 2>&1 &
done
wait
echo "✓ Concurrent requests completed"

# ============================================
# Step 7: 停止 profiling
# ============================================
echo ""
echo "=========================================="
echo "Step 7: 停止 profiling"
echo "=========================================="

curl -s -X POST http://localhost:${PORT}/stop_profile
echo ""
echo "✓ Profiling stopped"

# ============================================
# Step 8: 停止服务
# ============================================
echo ""
echo "=========================================="
echo "Step 8: 停止服务"
echo "=========================================="

kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null || true
sleep 5
echo "✓ Server stopped"

# ============================================
# Step 9: 检查输出
# ============================================
echo ""
echo "=========================================="
echo "Step 9: 检查 trace 文件"
echo "=========================================="

echo "Trace 目录: ${PROFILER_DIR}"
ls -lh "${PROFILER_DIR}/" 2>/dev/null || echo "(no trace files found)"

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "后续分析："
echo "  1. 用 Perfetto UI (https://ui.perfetto.dev/) 打开 trace 文件"
echo "  2. 观察 Prefill 步的 kernel 分布（GEMM 为主）"
echo "  3. 观察 Decode 步的 kernel 分布（attention 为主）"
echo "  4. 对比各组件耗时占比"
echo ""
echo "日志文件: ${LOG_FILE}"
