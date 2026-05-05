#!/usr/bin/env python3
"""Summarize CUDA work inside MLC BatchDecode NVTX ranges only."""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


KERNEL_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    ("gemv_gemm", tuple(re.compile(p, re.I) for p in (r"gemv", r"gemm", r"cublas"))),
    ("flash_attention", tuple(re.compile(p, re.I) for p in (r"flash", r"attention"))),
    ("gdn_linear", tuple(re.compile(p, re.I) for p in (r"gdn", r"delta", r"recurrent"))),
    (
        "sampling_logits",
        tuple(re.compile(p, re.I) for p in (r"softmax", r"sample", r"top_p", r"argmax", r"sort")),
    ),
    ("kv_cache", tuple(re.compile(p, re.I) for p in (r"kv", r"cache", r"copy_single_page", r"append"))),
    (
        "elementwise",
        tuple(
            re.compile(p, re.I)
            for p in (r"fused", r"reshape", r"take", r"transpose", r"slice", r"broadcast", r"cast")
        ),
    ),
)

RUNTIME_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    ("graph_launch", (re.compile(r"cudaGraphLaunch", re.I),)),
    ("kernel_launch", (re.compile(r"LaunchKernel|cuLaunchKernel", re.I),)),
    ("memcpy", (re.compile(r"Memcpy", re.I),)),
    ("sync", (re.compile(r"Synchronize|Wait|Event", re.I),)),
    ("alloc_free", (re.compile(r"Malloc|Free", re.I),)),
)


def classify(name: str, patterns: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]) -> str:
    for category, category_patterns in patterns:
        if any(pattern.search(name) for pattern in category_patterns):
            return category
    return "other"


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return bool(row and row[0])


def decode_ranges(con: sqlite3.Connection, pattern: str) -> list[tuple[int, int]]:
    rows = con.execute(
        """
        SELECT start, end
          FROM NVTX_EVENTS
         WHERE end IS NOT NULL AND text LIKE ?
         ORDER BY start
        """,
        (pattern,),
    ).fetchall()
    return [(int(start), int(end)) for start, end in rows]


def summarize_intervals(
    con: sqlite3.Connection,
    table: str,
    name_join_column: str,
    ranges: list[tuple[int, int]],
    category_patterns: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...],
) -> tuple[dict[str, tuple[int, float]], list[tuple[str, int, float, float]]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    if not ranges or not table_exists(con, table):
        return {}, []

    for start, end in ranges:
        rows = con.execute(
            f"""
            SELECT COALESCE(s.value, printf('id:%s', t.{name_join_column})) AS name,
                   t.start AS start,
                   t.end AS end
              FROM {table} t
              LEFT JOIN StringIds s ON s.id = t.{name_join_column}
             WHERE t.end > ? AND t.start < ?
            """,
            (start, end),
        ).fetchall()
        for name, item_start, item_end in rows:
            duration_ns = max(0, min(int(item_end), end) - max(int(item_start), start))
            if duration_ns == 0:
                continue
            duration_ms = duration_ns / 1e6
            category = classify(name or "", category_patterns)
            totals[category][0] += 1
            totals[category][1] += duration_ms
            by_name[name or ""][0] += 1
            by_name[name or ""][1] += duration_ms
            by_name[name or ""][2] = max(by_name[name or ""][2], duration_ns / 1e3)

    top_names = sorted(
        ((name, int(values[0]), values[1], values[2]) for name, values in by_name.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    return {category: (int(values[0]), values[1]) for category, values in totals.items()}, top_names


def print_category_table(
    title: str, summary: dict[str, tuple[int, float]], decode_tokens: int | None
) -> None:
    print(title)
    denom = max(decode_tokens or 0, 1)
    for category in sorted(summary):
        calls, total_ms = summary[category]
        suffix = ""
        if decode_tokens:
            suffix = f" {calls / denom:7.3f} calls/tok {total_ms * 1000 / denom:9.3f} us/tok"
        print(f"  {category:18s} {calls:8d} {total_ms:10.3f} ms{suffix}")


def print_top(title: str, rows: list[tuple[str, int, float, float]], top: int, decode_tokens: int | None) -> None:
    print(title)
    denom = max(decode_tokens or 0, 1)
    for name, calls, total_ms, max_us in rows[:top]:
        suffix = ""
        if decode_tokens:
            suffix = f" {calls / denom:7.3f} calls/tok {total_ms * 1000 / denom:9.3f} us/tok"
        print(f"  {total_ms:10.3f} ms {calls:8d} max {max_us:9.3f} us{suffix}  {name[:160]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--decode-range-like", default="BatchDecode num_seqs=%")
    parser.add_argument("--decode-tokens", type=int)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = sqlite3.connect(args.sqlite)
    ranges = decode_ranges(con, args.decode_range_like)
    total_range_ms = sum(end - start for start, end in ranges) / 1e6
    print(f"Report: {args.sqlite}")
    print(
        f"Decode ranges: count={len(ranges)} total_range_ms={total_range_ms:.3f} "
        f"pattern={args.decode_range_like!r}"
    )
    if args.decode_tokens:
        print(f"Decode range time per token: {total_range_ms * 1000 / args.decode_tokens:.3f} us/tok")

    kernel_summary, top_kernels = summarize_intervals(
        con, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName", ranges, KERNEL_PATTERNS
    )
    runtime_summary, top_runtime = summarize_intervals(
        con, "CUPTI_ACTIVITY_KIND_RUNTIME", "nameId", ranges, RUNTIME_PATTERNS
    )
    print_category_table("Kernel categories in decode ranges", kernel_summary, args.decode_tokens)
    print_top("Top kernels in decode ranges", top_kernels, args.top, args.decode_tokens)
    print_category_table("Runtime categories in decode ranges", runtime_summary, args.decode_tokens)
    print_top("Top runtime calls in decode ranges", top_runtime, min(args.top, 15), args.decode_tokens)


if __name__ == "__main__":
    main()
