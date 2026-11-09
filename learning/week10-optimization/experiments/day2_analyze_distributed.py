#!/usr/bin/env python3
"""Day 2: 对比分析 TP=1 vs TP=2 的 trace，重点看 NCCL 通信开销。

用法:
    .venv/bin/python learning/week10-optimization/experiments/day2_analyze_distributed.py \
        --tp1-dir learning/week10-optimization/results/day2_traces/tp1 \
        --tp2-dir learning/week10-optimization/results/day2_traces/tp2
"""

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict


def load_trace(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def classify_kernel(name: str) -> str:
    name_lower = name.lower()

    # NCCL 通信（Day 2 重点）
    if any(k in name_lower for k in [
        "nccl", "allreduce", "all_reduce", "allgather", "all_gather",
        "reducescatter", "reduce_scatter", "broadcast", "ncclkernel",
    ]):
        return "Communication (NCCL)"

    # FlashAttention（cutlass flash:: 在 GEMM 之前匹配）
    if any(k in name_lower for k in [
        "flash_attn", "flash_fwd", "flash_bwd",
        "flashattnfwd", "flashattncombine",
        "fmha", "sdpa", "flashinfer", "paged_attention",
    ]):
        return "Attention (FlashAttn)"
    if "cutlass" in name_lower and "flash::" in name:
        return "Attention (FlashAttn)"

    # KV Cache
    if "reshape_and_cache" in name_lower:
        return "KV Cache"

    # GEMM
    if any(k in name_lower for k in [
        "gemm", "cutlass", "cublas", "sm90", "sm80",
        "ampere", "hopper", "warp_specialized", "nvjet",
    ]):
        return "GEMM (Linear)"

    # Normalization
    if any(k in name_lower for k in ["layernorm", "rmsnorm", "layer_norm", "rms_norm"]):
        return "Normalization"

    # Activation
    if any(k in name_lower for k in ["silu", "gelu", "activation", "silu_and_mul"]):
        return "Activation"

    if "softmax" in name_lower:
        return "Softmax"

    if "embedding" in name_lower or "gather" in name_lower:
        return "Embedding"

    if any(k in name_lower for k in ["memcpy", "memset", "copy_"]):
        return "Memory Ops"

    if any(k in name_lower for k in ["elementwise", "add_kernel", "mul_kernel"]):
        return "Elementwise"

    return "Other"


def extract_cuda_kernels(trace_data: dict) -> list[dict]:
    """提取所有 CUDA kernel 事件"""
    events = trace_data.get("traceEvents", [])
    kernels = []
    for ev in events:
        if ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append({
                "name": ev.get("name", "unknown"),
                "dur_us": ev.get("dur", 0),
                "ts": ev.get("ts", 0),
            })
    return kernels


def summarize_by_component(kernels: list[dict]) -> dict:
    """按组件分类汇总"""
    summary = defaultdict(lambda: {"count": 0, "total_us": 0})
    for k in kernels:
        cat = classify_kernel(k["name"])
        summary[cat]["count"] += k.get("count", 1) if "count" in k else 1
        summary[cat]["total_us"] += k["dur_us"]
    return dict(summary)


def print_component_table(summary: dict, label: str):
    """打印组件级统计表"""
    total_us = sum(v["total_us"] for v in summary.values())
    if total_us == 0:
        print(f"  {label}: 无 CUDA kernel 数据")
        return

    sorted_items = sorted(summary.items(), key=lambda x: -x[1]["total_us"])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Total CUDA time: {total_us/1000:.1f} ms")
    print(f"{'='*60}")
    print(f"  {'Component':<25} {'Time(ms)':>10} {'%':>8} {'Count':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8}")
    for comp, stats in sorted_items:
        pct = stats["total_us"] / total_us * 100
        print(f"  {comp:<25} {stats['total_us']/1000:>10.2f} {pct:>7.1f}% {stats['count']:>8}")


def print_nccl_detail(kernels: list[dict], label: str):
    """打印 NCCL kernel 的详细信息"""
    nccl_kernels = defaultdict(lambda: {"count": 0, "total_us": 0})
    for k in kernels:
        if classify_kernel(k["name"]) == "Communication (NCCL)":
            short_name = k["name"][:80]
            nccl_kernels[short_name]["count"] += 1
            nccl_kernels[short_name]["total_us"] += k["dur_us"]

    if not nccl_kernels:
        print(f"\n  {label}: 无 NCCL kernel（符合预期，TP=1 无通信）")
        return

    print(f"\n  {label} — NCCL Kernel 明细:")
    print(f"  {'Kernel':<60} {'Time(ms)':>10} {'Count':>8} {'Avg(us)':>10}")
    print(f"  {'-'*60} {'-'*10} {'-'*8} {'-'*10}")
    for name, stats in sorted(nccl_kernels.items(), key=lambda x: -x[1]["total_us"]):
        avg = stats["total_us"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {name:<60} {stats['total_us']/1000:>10.3f} {stats['count']:>8} {avg:>10.1f}")


def find_traces(trace_dir: str, workers_only: bool = True) -> list[str]:
    """查找目录下所有 trace 文件。workers_only=True 时跳过 async_llm CPU trace。"""
    patterns = ["*.json.gz", "*.json", "*.pt.trace.json.gz", "*.pt.trace.json"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(trace_dir, p)))
    files = sorted(set(files))
    if workers_only:
        files = [f for f in files if "async_llm" not in os.path.basename(f)]
    return files


def main():
    parser = argparse.ArgumentParser(description="Day 2: TP=1 vs TP=2 分布式 Profiling 对比")
    parser.add_argument("--tp1-dir", required=True, help="TP=1 trace 目录")
    parser.add_argument("--tp2-dir", required=True, help="TP=2 trace 目录")
    parser.add_argument("--top-k", type=int, default=15, help="Top K kernels to show")
    args = parser.parse_args()

    # 加载 TP=1 traces
    tp1_files = find_traces(args.tp1_dir)
    tp2_files = find_traces(args.tp2_dir)

    print(f"TP=1 trace files: {len(tp1_files)}")
    for f in tp1_files:
        print(f"  {os.path.basename(f)}")
    print(f"TP=2 trace files: {len(tp2_files)}")
    for f in tp2_files:
        print(f"  {os.path.basename(f)}")

    if not tp1_files:
        print("ERROR: 未找到 TP=1 trace 文件")
        sys.exit(1)
    if not tp2_files:
        print("ERROR: 未找到 TP=2 trace 文件")
        sys.exit(1)

    # TP=1 分析（只有 1 个 rank）
    print("\n" + "=" * 60)
    print("  加载 TP=1 trace...")
    print("=" * 60)
    tp1_trace = load_trace(tp1_files[0])
    tp1_kernels = extract_cuda_kernels(tp1_trace)
    tp1_summary = summarize_by_component(tp1_kernels)
    print_component_table(tp1_summary, "TP=1 (单卡)")
    print_nccl_detail(tp1_kernels, "TP=1")

    # TP=2 分析（每个 rank 单独分析）
    for i, tf in enumerate(tp2_files):
        # 跳过 profiler_out 等文本文件
        if not (tf.endswith(".json") or tf.endswith(".json.gz")):
            continue
        print(f"\n{'='*60}")
        print(f"  加载 TP=2 rank {i} trace: {os.path.basename(tf)}")
        print("=" * 60)
        trace_data = load_trace(tf)
        kernels = extract_cuda_kernels(trace_data)
        summary = summarize_by_component(kernels)
        print_component_table(summary, f"TP=2 Rank {i}")
        print_nccl_detail(kernels, f"TP=2 Rank {i}")

    # 对比摘要
    tp1_total = sum(v["total_us"] for v in tp1_summary.values())
    tp1_nccl = tp1_summary.get("Communication (NCCL)", {}).get("total_us", 0)
    tp1_gemm = tp1_summary.get("GEMM (Linear)", {}).get("total_us", 0)
    tp1_attn = tp1_summary.get("Attention (FlashAttn)", {}).get("total_us", 0)

    # 取 TP=2 rank 0 做对比
    if tp2_files:
        tp2_trace = load_trace(tp2_files[0])
        tp2_kernels = extract_cuda_kernels(tp2_trace)
        tp2_summary = summarize_by_component(tp2_kernels)
        tp2_total = sum(v["total_us"] for v in tp2_summary.values())
        tp2_nccl = tp2_summary.get("Communication (NCCL)", {}).get("total_us", 0)
        tp2_gemm = tp2_summary.get("GEMM (Linear)", {}).get("total_us", 0)
        tp2_attn = tp2_summary.get("Attention (FlashAttn)", {}).get("total_us", 0)

        print(f"\n{'='*60}")
        print("  TP=1 vs TP=2 (Rank 0) 关键指标对比")
        print(f"{'='*60}")
        print(f"  {'Metric':<30} {'TP=1':>12} {'TP=2 R0':>12} {'变化':>10}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

        def fmt_row(label, v1, v2):
            if v1 > 0:
                change = (v2 - v1) / v1 * 100
                sign = "+" if change >= 0 else ""
                return f"  {label:<30} {v1/1000:>11.1f}ms {v2/1000:>11.1f}ms {sign}{change:>8.1f}%"
            return f"  {label:<30} {v1/1000:>11.1f}ms {v2/1000:>11.1f}ms {'N/A':>10}"

        print(fmt_row("Total CUDA Time", tp1_total, tp2_total))
        print(fmt_row("GEMM Time", tp1_gemm, tp2_gemm))
        print(fmt_row("FlashAttn Time", tp1_attn, tp2_attn))
        print(fmt_row("NCCL Communication", tp1_nccl, tp2_nccl))

        if tp2_total > 0:
            print(f"\n  TP=2 NCCL 通信占比: {tp2_nccl/tp2_total*100:.1f}%")
            print(f"  TP=2 GEMM 占比: {tp2_gemm/tp2_total*100:.1f}%")
            if tp1_gemm > 0:
                gemm_ratio = tp2_gemm / tp1_gemm
                print(f"  GEMM 缩减比: {gemm_ratio:.2f}x（理论 TP=2 应为 ~0.5x）")


if __name__ == "__main__":
    main()
