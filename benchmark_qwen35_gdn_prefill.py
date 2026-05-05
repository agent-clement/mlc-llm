#!/usr/bin/env python3
"""Benchmark and validate Qwen3.5 GDN prefill recurrence kernels.

This isolates the recurrent GatedDeltaNet prefill core.  It compares a
straightforward sequential recurrence against Flash Linear Attention's chunked
GatedDeltaRule for the Qwen3.5 0.8B dimensions used by the MLC experiments.
"""

from __future__ import annotations

import argparse
import math
import time

import torch


def sequential_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrence matching MLC's generic GDN prefill semantics.

    Shapes:
      q/k: [B, T, Hg, K], v/out: [B, T, H, V], g_log/beta: [B, T, H],
      initial_state/final_state: [B, H, K, V].
    """
    batch, seq_len, key_heads, key_dim = q.shape
    value_heads = v.shape[2]
    heads_per_group = value_heads // key_heads
    scale = key_dim**-0.5

    state = initial_state.float().clone()
    outputs = []
    for t in range(seq_len):
        per_head = []
        gate_t = torch.exp(g_log[:, t]).float()
        beta_t = beta[:, t].float()
        for h in range(value_heads):
            kh = h // heads_per_group
            # state_h is [B, K, V].
            state_h = state[:, h] * gate_t[:, h].view(batch, 1, 1)
            k_t = k[:, t, kh].float()
            q_t = q[:, t, kh].float()
            v_t = v[:, t, h].float()

            # Equivalent to dot(S[:, col], k[:]) for every V column.
            pred = torch.einsum("bkv,bk->bv", state_h, k_t)
            delta = beta_t[:, h].view(batch, 1) * (v_t - pred)
            state_h = state_h + k_t.unsqueeze(-1) * delta.unsqueeze(1)

            out_h = torch.einsum("bkv,bk->bv", state_h, q_t) * scale
            state[:, h] = state_h
            per_head.append(out_h)
        outputs.append(torch.stack(per_head, dim=1))

    return torch.stack(outputs, dim=1).to(q.dtype), state.to(initial_state.dtype)


def load_fla_chunk_gated_delta_rule():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore
    except Exception as err:  # pragma: no cover
        raise RuntimeError(
            "Could not import fla.ops.gated_delta_rule.chunk_gated_delta_rule. "
            "Install flash-linear-attention in the active environment."
        ) from err
    return chunk_gated_delta_rule


def cuda_time(fn, warmup: int, runs: int) -> tuple[float, object]:
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=276)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    q = torch.randn(
        args.batch, args.seq_len, args.key_heads, args.key_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(dtype)
    v = torch.randn(
        args.batch, args.seq_len, args.value_heads, args.value_dim, device=device, dtype=dtype
    )
    # Qwen3.5 gate before exponentiation: -exp(A_log) * softplus(alpha + dt_bias).
    # Use a realistic negative range without depending on model weights.
    g_log = -torch.nn.functional.softplus(
        torch.randn(args.batch, args.seq_len, args.value_heads, device=device, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(args.batch, args.seq_len, args.value_heads, device=device, dtype=torch.float32)
    )
    initial_state = torch.zeros(
        args.batch, args.value_heads, args.key_dim, args.value_dim, device=device, dtype=dtype
    )

    print(
        "shape "
        f"B={args.batch} T={args.seq_len} Hg={args.key_heads} H={args.value_heads} "
        f"K={args.key_dim} V={args.value_dim} dtype={args.dtype}"
    )

    chunk_gated_delta_rule = load_fla_chunk_gated_delta_rule()

    if not args.skip_reference:
        t0 = time.perf_counter()
        ref_out, ref_state = sequential_gdn(q, k, v, g_log, beta, initial_state)
        torch.cuda.synchronize()
        ref_seconds = time.perf_counter() - t0
        print(f"reference_seconds={ref_seconds:.6f}")
    else:
        ref_out = ref_state = None

    def run_fla():
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g_log,
            beta=beta,
            scale=1.0 / math.sqrt(args.key_dim),
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

    fla_seconds, (fla_out, fla_state) = cuda_time(
        run_fla, warmup=args.warmup_runs, runs=args.benchmark_runs
    )
    total_tokens = args.batch * args.seq_len * args.benchmark_runs
    print(
        "fla_chunk "
        f"total_seconds={fla_seconds:.6f} avg_seconds={fla_seconds / args.benchmark_runs:.6f} "
        f"tokens_per_second={total_tokens / fla_seconds:.3f}"
    )

    if ref_out is not None and ref_state is not None:
        out_diff = (fla_out - ref_out).abs()
        state_diff = (fla_state - ref_state).abs()
        print(
            "correctness "
            f"out_max_abs={out_diff.max().item():.6e} out_mean_abs={out_diff.mean().item():.6e} "
            f"state_max_abs={state_diff.max().item():.6e} "
            f"state_mean_abs={state_diff.mean().item():.6e}"
        )


if __name__ == "__main__":
    main()
