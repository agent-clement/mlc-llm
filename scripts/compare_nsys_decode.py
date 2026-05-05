#!/usr/bin/env python3
"""Compare two Nsight Systems SQLite exports for decode-focused profiling.

The output is intentionally small: it groups CUDA kernels and runtime calls into
categories that matter for the Qwen3.5 MLC-vLLM batch-1 comparison.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Category:
    name: str
    patterns: tuple[re.Pattern[str], ...]

    def matches(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


KERNEL_CATEGORIES = (
    Category("gemv_gemm", tuple(re.compile(p, re.I) for p in (r"gemv", r"gemm", r"cublas"))),
    Category("flash_attention", tuple(re.compile(p, re.I) for p in (r"flash", r"attention"))),
    Category("gdn_linear", tuple(re.compile(p, re.I) for p in (r"gdn", r"delta", r"recurrent"))),
    Category("sampling_logits", tuple(re.compile(p, re.I) for p in (r"softmax", r"sample", r"top_p", r"argmax", r"sort"))),
    Category("kv_cache", tuple(re.compile(p, re.I) for p in (r"kv", r"cache", r"copy_single_page", r"append"))),
    Category("elementwise", tuple(re.compile(p, re.I) for p in (r"fused", r"reshape", r"take", r"transpose", r"slice", r"broadcast", r"cast"))),
)

RUNTIME_CATEGORIES = (
    Category("graph_launch", (re.compile(r"cudaGraphLaunch", re.I),)),
    Category("kernel_launch", (re.compile(r"LaunchKernel|cuLaunchKernel", re.I),)),
    Category("memcpy", (re.compile(r"Memcpy", re.I),)),
    Category("sync", (re.compile(r"Synchronize|Wait|Event", re.I),)),
    Category("alloc_free", (re.compile(r"Malloc|Free", re.I),)),
)


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "select count(*) from sqlite_master where type = 'table' and name = ?", (name,)
    ).fetchone()
    return bool(row and row[0])


def classify(name: str, categories: tuple[Category, ...]) -> str:
    for category in categories:
        if category.matches(name):
            return category.name
    return "other"


def summarize_kernels(con: sqlite3.Connection, tokens: int) -> dict[str, tuple[int, float]]:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return {}
    rows = con.execute(
        """
        select coalesce(s.value, printf('id:%s', k.demangledName)) name,
               count(*) calls,
               sum(k.end - k.start) / 1e6 total_ms
          from CUPTI_ACTIVITY_KIND_KERNEL k
          left join StringIds s on s.id = k.demangledName
         group by name
        """
    ).fetchall()
    result: dict[str, tuple[int, float]] = {}
    for name, calls, total_ms in rows:
        category = classify(name or "", KERNEL_CATEGORIES)
        prev_calls, prev_ms = result.get(category, (0, 0.0))
        result[category] = (prev_calls + calls, prev_ms + total_ms)
    return normalize(result, tokens)


def summarize_runtime(con: sqlite3.Connection, tokens: int) -> dict[str, tuple[int, float]]:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return {}
    rows = con.execute(
        """
        select coalesce(s.value, printf('id:%s', r.nameId)) name,
               count(*) calls,
               sum(r.end - r.start) / 1e6 total_ms
          from CUPTI_ACTIVITY_KIND_RUNTIME r
          left join StringIds s on s.id = r.nameId
         group by name
        """
    ).fetchall()
    result: dict[str, tuple[int, float]] = {}
    for name, calls, total_ms in rows:
        category = classify(name or "", RUNTIME_CATEGORIES)
        prev_calls, prev_ms = result.get(category, (0, 0.0))
        result[category] = (prev_calls + calls, prev_ms + total_ms)
    return normalize(result, tokens)


def normalize(summary: dict[str, tuple[int, float]], tokens: int) -> dict[str, tuple[int, float]]:
    denom = max(tokens, 1)
    return {category: (calls / denom, total_ms * 1000 / denom) for category, (calls, total_ms) in summary.items()}


def print_table(title: str, left_name: str, right_name: str, left: dict[str, tuple[int, float]], right: dict[str, tuple[int, float]]) -> None:
    print(title)
    print(f"{'category':18s} {left_name:>24s} {right_name:>24s} {'delta_us/tok':>14s}")
    categories = sorted(set(left) | set(right))
    for category in categories:
        l_calls, l_us = left.get(category, (0.0, 0.0))
        r_calls, r_us = right.get(category, (0.0, 0.0))
        print(
            f"{category:18s} "
            f"{l_us:8.3f} us {l_calls:7.3f} c "
            f"{r_us:8.3f} us {r_calls:7.3f} c "
            f"{l_us - r_us:14.3f}"
        )
    print()


def infer_decode_tokens(con: sqlite3.Connection, label: str) -> int:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise ValueError(f"Cannot infer token count for {label}: no kernel table")
    rows = con.execute(
        """
        select coalesce(s.value, printf('id:%s', k.demangledName)) name,
               count(*) calls,
               sum(k.end - k.start) / 1e6 total_ms,
               avg(k.end - k.start) / 1e3 avg_us
          from CUPTI_ACTIVITY_KIND_KERNEL k
          left join StringIds s on s.id = k.demangledName
         group by name
        """
    ).fetchall()
    candidates = []
    for name, calls, total_ms, avg_us in rows:
        text = name or ""
        if re.search(r"gemv|gemvx|cublas", text, re.I) and avg_us >= 100:
            candidates.append((float(total_ms), int(calls), text))
    if not candidates:
        graph_launches = infer_tokens_from_graph_launches(con, label)
        if graph_launches is not None:
            return graph_launches
        raise ValueError(
            f"Cannot infer token count for {label}: no dominant GEMV/cublas decode kernel found"
        )
    _, calls, name = max(candidates)
    print(f"inferred_{label}_tokens={calls} from_kernel={name[:100]}")
    return calls


def infer_tokens_from_graph_launches(con: sqlite3.Connection, label: str) -> int | None:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return None
    row = con.execute(
        """
        select count(*)
          from CUPTI_ACTIVITY_KIND_RUNTIME r
          left join StringIds s on s.id = r.nameId
         where coalesce(s.value, printf('id:%s', r.nameId)) like '%cudaGraphLaunch%'
        """
    ).fetchone()
    calls = int(row[0]) if row is not None else 0
    if calls <= 0:
        return None
    print(f"inferred_{label}_tokens={calls} from_runtime=cudaGraphLaunch")
    return calls


def resolve_tokens(value: str, con: sqlite3.Connection, label: str) -> int:
    if value == "auto":
        return infer_decode_tokens(con, label)
    tokens = int(value)
    if tokens <= 0:
        raise ValueError(f"{label} token count must be positive")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True, help="Left Nsight SQLite export")
    parser.add_argument("--right", type=Path, required=True, help="Right Nsight SQLite export")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--left-tokens", default="auto")
    parser.add_argument("--right-tokens", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = sqlite3.connect(args.left)
    right = sqlite3.connect(args.right)
    left_tokens = resolve_tokens(args.left_tokens, left, args.left_name)
    right_tokens = resolve_tokens(args.right_tokens, right, args.right_name)
    print_table(
        "Kernel categories, normalized per decode token",
        args.left_name,
        args.right_name,
        summarize_kernels(left, left_tokens),
        summarize_kernels(right, right_tokens),
    )
    print_table(
        "CUDA runtime categories, normalized per decode token",
        args.left_name,
        args.right_name,
        summarize_runtime(left, left_tokens),
        summarize_runtime(right, right_tokens),
    )


if __name__ == "__main__":
    main()
