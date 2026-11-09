#!/bin/bash
# PD 分离实验（NIXL）：启动 Prefill 端
# GPU 0, port 8100, kv_role=kv_producer

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /data/model/Qwen2.5-7B \
    --port 8100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_rank": 0, "kv_parallel_size": 2}' \
    --max-model-len 4096
