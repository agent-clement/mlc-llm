#!/usr/bin/env python3
"""Benchmark vLLM's bundled paged FlashAttention backend on Qwen3.5-like shapes.

Run this with the working vLLM conda environment, outside the vLLM source tree if
possible:

  /home/cwong/Projects/miniconda/envs/vllm/bin/python scripts/benchmark_vllm_flash_attn_paged.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    is_fa_version_supported,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-len", type=int, default=320)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-query-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=50)
    parser.add_argument("--benchmark-runs", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not is_fa_version_supported(args.fa_version):
        raise SystemExit(f"FA{args.fa_version} is not supported in this environment")

    torch.set_default_device("cuda")
    dtype = getattr(torch, args.dtype)
    max_blocks = math.ceil(args.context_len / args.block_size)
    num_blocks = max_blocks
    scale = args.head_dim**-0.5

    q = torch.randn(1, args.num_query_heads, args.head_dim, dtype=dtype)
    k = torch.randn(num_blocks, args.block_size, args.num_kv_heads, args.head_dim, dtype=dtype)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seqused_k = torch.tensor([args.context_len], dtype=torch.int32, device="cuda")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, num_blocks)
    out = torch.empty_like(q)

    for _ in range(args.warmup_runs):
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=args.context_len,
            softmax_scale=scale,
            causal=True,
            block_table=block_table,
            fa_version=args.fa_version,
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(args.benchmark_runs):
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=args.context_len,
            softmax_scale=scale,
            causal=True,
            block_table=block_table,
            fa_version=args.fa_version,
        )
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    cuda_ms = start_event.elapsed_time(end_event)
    print(
        "vllm_flash_attn_paged "
        f"fa_version={args.fa_version} dtype={args.dtype} "
        f"context_len={args.context_len} block_size={args.block_size} "
        f"q_heads={args.num_query_heads} kv_heads={args.num_kv_heads} head_dim={args.head_dim} "
        f"runs={args.benchmark_runs} cuda_us_per_call={cuda_ms * 1000 / args.benchmark_runs:.3f} "
        f"wall_us_per_call={wall_seconds * 1e6 / args.benchmark_runs:.3f}"
    )


if __name__ == "__main__":
    main()
