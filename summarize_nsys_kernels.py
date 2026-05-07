#!/usr/bin/env python3
"""Summarize CUDA activity from an Nsight Systems SQLite export."""

import argparse
import re
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    name: str
    patterns: tuple[re.Pattern[str], ...]

    def matches(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


KERNEL_CATEGORIES = (
    Category("gemv_gemm", tuple(re.compile(p, re.I) for p in (r"gemv", r"gemm", r"cublas"))),
    Category("flash_attention", tuple(re.compile(p, re.I) for p in (r"flash", r"attention"))),
    Category("gdn_linear", tuple(re.compile(p, re.I) for p in (r"gdn", r"delta", r"recurrent"))),
    Category(
        "sampling_logits",
        tuple(re.compile(p, re.I) for p in (r"softmax", r"sample", r"top_p", r"argmax", r"sort")),
    ),
    Category("kv_cache", tuple(re.compile(p, re.I) for p in (r"kv", r"cache", r"copy_single_page", r"append"))),
    Category(
        "elementwise",
        tuple(re.compile(p, re.I) for p in (r"fused", r"reshape", r"take", r"transpose", r"slice", r"broadcast", r"cast")),
    ),
)


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    return (
        cur.execute(
            "select count(*) from sqlite_master where type = 'table' and name = ?", [name]
        ).fetchone()[0]
        > 0
    )


def print_total(cur: sqlite3.Cursor, table: str, label: str, tokens: int) -> None:
    if not table_exists(cur, table):
        return
    total_ms, calls = cur.execute(
        f"select coalesce(sum(end - start), 0) / 1e6, count(*) from {table}"
    ).fetchone()
    print(
        f"total_{label}_ms={total_ms:.3f} calls={calls} "
        f"per_unit_us={total_ms * 1000 / max(tokens, 1):.3f} "
        f"calls_per_unit={calls / max(tokens, 1):.3f}"
    )


def classify_kernel(name: str) -> str:
    for category in KERNEL_CATEGORIES:
        if category.matches(name):
            return category.name
    return "other"


def print_kernel_categories(cur: sqlite3.Cursor, tokens: int) -> None:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return
    rows = cur.execute(
        """
        select coalesce(s.value, printf('id:%s', k.demangledName)) name,
               count(*) calls,
               sum(k.end - k.start) / 1e6 total_ms
          from CUPTI_ACTIVITY_KIND_KERNEL k
          left join StringIds s on s.id = k.demangledName
         group by name
        """
    ).fetchall()
    summary: dict[str, tuple[int, float]] = {}
    for name, calls, total_ms in rows:
        category = classify_kernel(name or "")
        prev_calls, prev_ms = summary.get(category, (0, 0.0))
        summary[category] = (prev_calls + int(calls), prev_ms + float(total_ms))

    print("\nKernel categories:")
    for category, (calls, total_ms) in sorted(
        summary.items(), key=lambda item: item[1][1], reverse=True
    ):
        print(
            f"{total_ms:9.3f} ms {calls:6d} calls "
            f"{total_ms * 1000 / max(tokens, 1):9.3f} us/unit "
            f"{calls / max(tokens, 1):7.3f} calls/unit {category}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="Path produced by `nsys export --type sqlite ...`.")
    parser.add_argument("--tokens", type=int, default=1, help="Decode tokens or phase runs.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--categories", action="store_true", help="Print candidate-oriented kernel categories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    print_total(cur, "CUPTI_ACTIVITY_KIND_KERNEL", "kernel", args.tokens)
    print_total(cur, "CUPTI_ACTIVITY_KIND_RUNTIME", "cuda_runtime_api", args.tokens)
    print_total(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy", args.tokens)
    print_total(cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "sync", args.tokens)
    print_total(cur, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "graph_trace", args.tokens)
    print_total(cur, "CUDA_GRAPH_EVENTS", "cuda_graph_event", args.tokens)
    if args.categories:
        print_kernel_categories(cur, args.tokens)

    query = """
        select coalesce(s.value, printf('id:%s', k.demangledName)) name,
               count(*) calls,
               sum(k.end - k.start) / 1e6 total_ms,
               avg(k.end - k.start) / 1e3 avg_us
          from CUPTI_ACTIVITY_KIND_KERNEL k
          left join StringIds s on s.id = k.demangledName
         group by name
         order by total_ms desc
         limit ?
    """
    print("\nTop kernels:")
    for name, calls, kernel_ms, avg_us in cur.execute(query, [args.limit]):
        per_unit_us = kernel_ms * 1000 / max(args.tokens, 1)
        print(
            f"{kernel_ms:9.3f} ms {calls:6d} calls {avg_us:9.3f} us avg "
            f"{per_unit_us:9.3f} us/unit {calls / max(args.tokens, 1):7.3f} calls/unit "
            f"{name[:120]}"
        )

    if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        query = """
            select coalesce(s.value, printf('id:%s', r.nameId)) name,
                   count(*) calls,
                   sum(r.end - r.start) / 1e6 total_ms,
                   avg(r.end - r.start) / 1e3 avg_us
              from CUPTI_ACTIVITY_KIND_RUNTIME r
              left join StringIds s on s.id = r.nameId
             group by name
             order by total_ms desc
             limit ?
        """
        print("\nTop CUDA runtime APIs:")
        for name, calls, total_ms, avg_us in cur.execute(query, [args.limit]):
            per_unit_us = total_ms * 1000 / max(args.tokens, 1)
            print(
                f"{total_ms:9.3f} ms {calls:6d} calls {avg_us:9.3f} us avg "
                f"{per_unit_us:9.3f} us/unit {calls / max(args.tokens, 1):7.3f} calls/unit "
                f"{name[:120]}"
            )

    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        query = """
            select copyKind,
                   count(*) calls,
                   sum(bytes) bytes,
                   sum(end - start) / 1e6 total_ms,
                   avg(end - start) / 1e3 avg_us
              from CUPTI_ACTIVITY_KIND_MEMCPY
             group by copyKind
             order by total_ms desc
             limit ?
        """
        print("\nMemcpy by kind:")
        for copy_kind, calls, num_bytes, total_ms, avg_us in cur.execute(query, [args.limit]):
            per_unit_us = total_ms * 1000 / max(args.tokens, 1)
            print(
                f"{total_ms:9.3f} ms {calls:6d} calls {avg_us:9.3f} us avg "
                f"{per_unit_us:9.3f} us/unit {num_bytes / 1024 / 1024:9.3f} MiB "
                f"copyKind={copy_kind}"
            )


if __name__ == "__main__":
    main()
