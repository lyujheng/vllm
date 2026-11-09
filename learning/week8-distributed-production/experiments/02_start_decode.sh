#!/bin/bash
# PD 分离实验（NIXL）：启动 Decode 端
# GPU 1, port 8200, kv_role=kv_consumer

# 同机部署需要避开 P 端的 NIXL 握手端口（默认 5600）
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model /data/model/Qwen2.5-7B \
    --port 8200 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_rank": 1, "kv_parallel_size": 2}' \
    --max-model-len 4096
