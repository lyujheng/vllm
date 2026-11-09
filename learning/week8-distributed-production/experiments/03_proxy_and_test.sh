#!/bin/bash
# PD 分离实验（NIXL）：启动 Proxy + 发送测试请求
#
# 使用方法：
#   1. 先在终端 1 运行: bash 01_start_prefill.sh
#   2. 等 P 端启动完成（看到 "Uvicorn running on ..."）
#   3. 在终端 2 运行: bash 02_start_decode.sh
#   4. 等 D 端启动完成
#   5. 在终端 3 运行: bash 03_proxy_and_test.sh

set -e

MODEL="/data/model/Qwen2.5-7B"
PREFILL_HOST="localhost:8100"
DECODE_HOST="localhost:8200"
PROXY_PORT=8000

# 结果保存目录
RESULT_DIR="/home/lvzheng/vllm/results/week8_pd_disagg"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULT_DIR/nixl_test_${TIMESTAMP}.log"

# 同时输出到终端和文件
exec > >(tee "$RESULT_FILE") 2>&1
echo "结果将保存到: $RESULT_FILE"

echo "=============================="
echo "检查 P 端和 D 端是否就绪..."
echo "=============================="

# 检查 P 端
if ! curl -s http://$PREFILL_HOST/v1/models > /dev/null 2>&1; then
    echo "错误：P 端 ($PREFILL_HOST) 未就绪，请先启动 01_start_prefill.sh"
    exit 1
fi
echo "P 端就绪: $PREFILL_HOST"

# 检查 D 端
if ! curl -s http://$DECODE_HOST/v1/models > /dev/null 2>&1; then
    echo "错误：D 端 ($DECODE_HOST) 未就绪，请先启动 02_start_decode.sh"
    exit 1
fi
echo "D 端就绪: $DECODE_HOST"

echo ""
echo "=============================="
echo "启动 Proxy (port $PROXY_PORT)..."
echo "=============================="

# 后台启动 Proxy
python3 /home/lvzheng/vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py \
    --model $MODEL \
    --prefill $PREFILL_HOST \
    --decode $DECODE_HOST \
    --port $PROXY_PORT &
PROXY_PID=$!

# 等 Proxy 启动
sleep 3

# 检查 Proxy 是否存活
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "错误：Proxy 启动失败"
    exit 1
fi
echo "Proxy 就绪: localhost:$PROXY_PORT (PID: $PROXY_PID)"

echo ""
echo "=============================="
echo "实验 1：单请求测试"
echo "=============================="

echo "发送请求到 Proxy..."
curl -s http://localhost:$PROXY_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0
    }' | python3 -m json.tool

echo ""
echo "=============================="
echo "实验 2：多请求顺序测试"
echo "=============================="

for i in 1 2 3; do
    echo "--- 请求 $i ---"
    curl -s http://localhost:$PROXY_PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "prompt": "Request '$i': Tell me about",
            "max_tokens": 30,
            "temperature": 0
        }' | python3 -m json.tool
    echo ""
done

echo ""
echo "=============================="
echo "实验 3：长 prompt 测试（验证较大 KV 传输）"
echo "=============================="

LONG_PROMPT=$(python3 -c "print('Hello ' * 200 + 'The answer is')")

curl -s http://localhost:$PROXY_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "'"$LONG_PROMPT"'",
        "max_tokens": 20,
        "temperature": 0
    }' | python3 -m json.tool

echo ""
echo "=============================="
echo "实验完成，清理 Proxy..."
echo "=============================="
kill $PROXY_PID 2>/dev/null
echo "Proxy 已停止"
echo ""
echo "P 端和 D 端仍在运行，可手动 Ctrl+C 停止"
