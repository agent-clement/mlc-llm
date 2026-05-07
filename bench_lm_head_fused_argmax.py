#!/usr/bin/env python3
"""Prototype fused LM-head matvec + argmax for Qwen3.5.

This is a standalone microbenchmark. It does not participate in MLC runtime.
"""

import argparse
import time

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


@triton.jit
def _matvec_block_argmax_kernel(weight, hidden, block_vals, block_idxs, vocab: tl.constexpr,
                                hidden_size: tl.constexpr, block_m: tl.constexpr,
                                block_k: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * block_m + tl.arange(0, block_m)
    offs_k = tl.arange(0, block_k)
    acc = tl.zeros((block_m,), dtype=tl.float32)
    for k0 in range(0, hidden_size, block_k):
        k = k0 + offs_k
        w = tl.load(
            weight + offs_m[:, None] * hidden_size + k[None, :],
            mask=(offs_m[:, None] < vocab) & (k[None, :] < hidden_size),
            other=0.0,
        )
        h = tl.load(hidden + k, mask=k < hidden_size, other=0.0)
        acc += tl.sum(w.to(tl.float32) * h[None, :].to(tl.float32), axis=1)

    acc = tl.where(offs_m < vocab, acc, -float("inf"))
    max_val = tl.max(acc, axis=0)
    max_idx = tl.min(tl.where(acc == max_val, offs_m, vocab), axis=0)
    tl.store(block_vals + pid, max_val)
    tl.store(block_idxs + pid, max_idx)


def load_weight(model_id: str, device: str):
    model_path = snapshot_download(repo_id=model_id)
    state = load_file(f"{model_path}/model.safetensors", device="cpu")
    weight_key = next(key for key in state if key.endswith("embed_tokens.weight"))
    weight = state[weight_key].contiguous().to(device=device, dtype=torch.float16)
    return weight


def bench_cuda_event(fn, warmup: int, runs: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="familiar-ai/logos-multitask-qwen3.5-2026-05-03-best")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(0)
    weight = load_weight(args.model, "cuda")
    vocab, hidden_size = weight.shape
    hidden = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    num_blocks = triton.cdiv(vocab, args.block_m)
    block_vals = torch.empty((num_blocks,), device="cuda", dtype=torch.float32)
    block_idxs = torch.empty((num_blocks,), device="cuda", dtype=torch.int64)

    def fused():
        _matvec_block_argmax_kernel[(num_blocks,)](
            weight,
            hidden,
            block_vals,
            block_idxs,
            vocab,
            hidden_size,
            block_m=args.block_m,
            block_k=args.block_k,
            num_warps=8,
        )
        best_block = torch.argmax(block_vals)
        return block_idxs[best_block]

    def torch_logits_argmax():
        logits = torch.matmul(weight.float(), hidden.float())
        return torch.argmax(logits)

    fused()
    torch_logits_argmax()
    torch.cuda.synchronize()

    fused_seconds = bench_cuda_event(fused, args.warmup_runs, args.runs)
    torch_seconds = bench_cuda_event(torch_logits_argmax, args.warmup_runs, args.runs)
    print(
        f"vocab={vocab} hidden={hidden_size} block_m={args.block_m} "
        f"fused_ms_per_run={1000 * fused_seconds / args.runs:.3f} "
        f"fused_runs_per_second={args.runs / fused_seconds:.3f} "
        f"torch_logits_argmax_ms_per_run={1000 * torch_seconds / args.runs:.3f} "
        f"torch_logits_argmax_runs_per_second={args.runs / torch_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
