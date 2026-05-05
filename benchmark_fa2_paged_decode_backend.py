#!/usr/bin/env python3
"""Microbenchmark the TVM FA2-style paged decode backend registration."""

import argparse
import time

import numpy as np
import tvm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-q-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=320)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--benchmark-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=["mlc", "vllm"], default="mlc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = tvm.cuda(0)
    if not device.exist:
        raise RuntimeError("CUDA device is not available.")
    backend = tvm.get_global_func("tvm.contrib.flash_attn.fa2_paged_decode", allow_missing=True)
    if backend is None:
        raise RuntimeError("tvm.contrib.flash_attn.fa2_paged_decode is not registered.")
    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("--num-q-heads must be divisible by --num-kv-heads")

    rng = np.random.default_rng(args.seed)
    num_pages = (args.context_len + args.page_size - 1) // args.page_size
    q = tvm.runtime.tensor(
        rng.normal(size=(1, args.num_q_heads, args.head_dim)).astype("float16"), device
    )
    k_mlc = rng.normal(
        size=(num_pages, args.page_size, args.num_kv_heads, args.head_dim)
    ).astype("float16")
    v_mlc = rng.normal(
        size=(num_pages, args.page_size, args.num_kv_heads, args.head_dim)
    ).astype("float16")
    if args.layout == "vllm":
        x = 8
        k_np = (
            k_mlc.reshape(num_pages, args.page_size, args.num_kv_heads, args.head_dim // x, x)
            .transpose(0, 2, 3, 1, 4)
            .copy()
        )
        v_np = v_mlc.transpose(0, 2, 3, 1).copy()
    else:
        k_np = k_mlc
        v_np = v_mlc
    k_pages = tvm.runtime.tensor(k_np, device)
    v_pages = tvm.runtime.tensor(v_np, device)
    block_table = tvm.runtime.tensor(np.arange(num_pages, dtype="int32").reshape(1, -1), device)
    seqused_k = tvm.runtime.tensor(np.array([args.context_len], dtype="int32"), device)
    cu_seqlens_q = tvm.runtime.tensor(np.array([0, 1], dtype="int32"), device)
    q_rope_position = tvm.runtime.tensor(np.array([args.context_len - 1], dtype="int32"), device)
    out = tvm.runtime.empty((1, args.num_q_heads, args.head_dim), "float16", device)
    lse = tvm.runtime.empty((1, args.num_q_heads), "float32", device)
    sm_scale = args.head_dim**-0.5

    def run_once() -> None:
        backend(
            q,
            k_pages,
            v_pages,
            block_table,
            seqused_k,
            cu_seqlens_q,
            q_rope_position,
            0,
            0,
            sm_scale,
            out,
            lse,
        )

    for _ in range(max(0, args.warmup_runs)):
        run_once()
    device.sync()

    start = time.perf_counter()
    for _ in range(max(1, args.benchmark_runs)):
        run_once()
    device.sync()
    elapsed = time.perf_counter() - start
    runs = max(1, args.benchmark_runs)
    print(
        "fa2_paged_decode_backend "
        f"num_q_heads={args.num_q_heads} "
        f"num_kv_heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} "
        f"page_size={args.page_size} "
        f"context_len={args.context_len} "
        f"layout={args.layout} "
        f"runs={runs} "
        f"total_seconds={elapsed:.6f} "
        f"seconds_per_call={elapsed / runs:.9f} "
        f"calls_per_second={runs / elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
