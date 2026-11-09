#!/usr/bin/env python3
"""Day 1: 解析 torch profiler trace，输出 kernel 耗时统计和组件级分解。

用法:
    cd /tmp
    .venv/bin/python /home/lvzheng/vllm/learning/week10-optimization/experiments/day1_analyze_trace.py \
        --trace-dir /home/lvzheng/vllm/learning/week10-optimization/results/day1_traces

输出:
    1. Top 20 CUDA kernels（按 CUDA time 排序）
    2. 按组件分类的耗时统计（Attention / FFN / Unembed / 其他）
    3. CPU time vs CUDA time 对比
"""

import argparse
import glob
import json
import gzip
import os
import sys
from collections import defaultdict


def load_trace(trace_path: str) -> dict:
    """加载 trace 文件（支持 .json 和 .json.gz）"""
    if trace_path.endswith(".gz"):
        with gzip.open(trace_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(trace_path, "r", encoding="utf-8") as f:
            return json.load(f)


def classify_kernel(name: str) -> str:
    """根据 kernel 名称分类到组件"""
    name_lower = name.lower()

    # Attention 相关（必须在 GEMM 之前匹配，因为 FlashAttention kernel
    # 名包含 cutlass/sm90 等关键词）
    if any(k in name_lower for k in [
        "flash_attn", "flash_fwd", "flash_bwd",
        "flashattnfwd", "flashattncombine",
        "fmha", "sdpa",
        "flashinfer", "paged_attention",
    ]):
        return "Attention (FlashAttn)"

    # cutlass flash:: 命名空间的也是 FlashAttention
    if "cutlass" in name_lower and "flash::" in name:
        return "Attention (FlashAttn)"

    # QKV / Output 投影（attention 的线性层）
    if any(k in name_lower for k in ["qkv", "q_proj", "k_proj", "v_proj", "o_proj"]):
        return "Attention (Projection)"

    # KV Cache 写入
    if "reshape_and_cache" in name_lower:
        return "KV Cache"

    # GEMM / CUTLASS / cuBLAS（通用矩阵乘）
    if any(k in name_lower for k in [
        "gemm", "cutlass", "cublas", "sm90", "sm80",
        "ampere", "hopper", "warp_specialized", "nvjet",
    ]):
        return "GEMM (Linear)"

    # LayerNorm / RMSNorm
    if any(k in name_lower for k in ["layernorm", "rmsnorm", "layer_norm", "rms_norm"]):
        return "Normalization"

    # Activation（SiLU, GELU, etc.）
    if any(k in name_lower for k in [
        "silu", "gelu", "activation", "silu_and_mul",
    ]):
        return "Activation"

    # Softmax
    if "softmax" in name_lower:
        return "Softmax"

    # Embedding
    if "embedding" in name_lower or "gather" in name_lower:
        return "Embedding"

    # NCCL 通信
    if "nccl" in name_lower or "allreduce" in name_lower or "allgather" in name_lower:
        return "Communication (NCCL)"

    # Memory operations
    if any(k in name_lower for k in ["memcpy", "memset", "copy_"]):
        return "Memory Ops"

    # Elementwise
    if any(k in name_lower for k in ["elementwise", "add_kernel", "mul_kernel"]):
        return "Elementwise"

    return "Other"


def analyze_trace_events(trace_data: dict) -> dict:
    """从 trace events 中提取 kernel 统计信息"""
    events = trace_data.get("traceEvents", [])

    kernel_stats = defaultdict(lambda: {"count": 0, "total_dur_us": 0})

    for event in events:
        # CUDA kernel events
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)  # microseconds
            kernel_stats[name]["count"] += 1
            kernel_stats[name]["total_dur_us"] += dur

        # cpu_op 或 user_annotation
        elif event.get("cat") == "cpu_op":
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)
            if dur > 0:
                kernel_stats[f"[CPU] {name}"]["count"] += 1
                kernel_stats[f"[CPU] {name}"]["total_dur_us"] += dur

    return dict(kernel_stats)


def print_top_kernels(kernel_stats: dict, top_n: int = 20):
    """打印 Top N CUDA kernels"""
    # 只看 GPU kernels（排除 [CPU] 前缀）
    gpu_kernels = {
        k: v for k, v in kernel_stats.items() if not k.startswith("[CPU]")
    }

    sorted_kernels = sorted(
        gpu_kernels.items(), key=lambda x: x[1]["total_dur_us"], reverse=True
    )

    total_gpu_time = sum(v["total_dur_us"] for v in gpu_kernels.values())

    print(f"\n{'='*80}")
    print(f"Top {top_n} CUDA Kernels (by total GPU time)")
    print(f"{'='*80}")
    print(f"Total GPU time: {total_gpu_time/1000:.2f} ms\n")
    print(f"{'Rank':<5} {'GPU Time (ms)':<15} {'%':>6} {'Count':<8} {'Kernel Name'}")
    print(f"{'-'*5} {'-'*15} {'-'*6} {'-'*8} {'-'*40}")

    for i, (name, stats) in enumerate(sorted_kernels[:top_n]):
        time_ms = stats["total_dur_us"] / 1000
        pct = stats["total_dur_us"] / total_gpu_time * 100 if total_gpu_time > 0 else 0
        # 截断过长的 kernel 名
        short_name = name[:80] + "..." if len(name) > 80 else name
        print(f"{i+1:<5} {time_ms:<15.3f} {pct:>5.1f}% {stats['count']:<8} {short_name}")


def print_component_breakdown(kernel_stats: dict):
    """按组件分类打印耗时分解"""
    gpu_kernels = {
        k: v for k, v in kernel_stats.items() if not k.startswith("[CPU]")
    }

    component_stats = defaultdict(lambda: {"count": 0, "total_dur_us": 0})
    for name, stats in gpu_kernels.items():
        component = classify_kernel(name)
        component_stats[component]["count"] += stats["count"]
        component_stats[component]["total_dur_us"] += stats["total_dur_us"]

    total_gpu_time = sum(v["total_dur_us"] for v in gpu_kernels.values())

    sorted_components = sorted(
        component_stats.items(), key=lambda x: x[1]["total_dur_us"], reverse=True
    )

    print(f"\n{'='*80}")
    print(f"Component-level Breakdown")
    print(f"{'='*80}")
    print(f"Total GPU time: {total_gpu_time/1000:.2f} ms\n")
    print(f"{'Component':<30} {'GPU Time (ms)':<15} {'%':>6} {'Count':<8}")
    print(f"{'-'*30} {'-'*15} {'-'*6} {'-'*8}")

    for component, stats in sorted_components:
        time_ms = stats["total_dur_us"] / 1000
        pct = stats["total_dur_us"] / total_gpu_time * 100 if total_gpu_time > 0 else 0
        print(f"{component:<30} {time_ms:<15.3f} {pct:>5.1f}% {stats['count']:<8}")


def main():
    parser = argparse.ArgumentParser(description="Analyze torch profiler traces")
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing trace files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top kernels to show (default: 20)",
    )
    args = parser.parse_args()

    # 查找所有 trace 文件
    trace_files = sorted(
        glob.glob(os.path.join(args.trace_dir, "**/*.json*"), recursive=True)
    )

    if not trace_files:
        print(f"No trace files found in {args.trace_dir}")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace file(s):")
    for f in trace_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")

    # 逐个分析
    for trace_file in trace_files:
        print(f"\n{'#'*80}")
        print(f"# Analyzing: {os.path.basename(trace_file)}")
        print(f"{'#'*80}")

        try:
            trace_data = load_trace(trace_file)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        kernel_stats = analyze_trace_events(trace_data)

        if not kernel_stats:
            print("  No kernel events found in trace.")
            continue

        print_top_kernels(kernel_stats, args.top_n)
        print_component_breakdown(kernel_stats)


if __name__ == "__main__":
    main()
