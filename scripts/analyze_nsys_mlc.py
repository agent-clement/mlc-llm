#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export for MLC decode profiling."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def ms(expr: str) -> str:
    return f"({expr}) / 1e6"


def fetch_one(con: sqlite3.Connection, query: str, args: tuple[object, ...] = ()) -> sqlite3.Row:
    row = con.execute(query, args).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no rows: {query}")
    return row


def print_runtime_summary(con: sqlite3.Connection, decode_tokens: int | None) -> None:
    selected = [
        "cudaGraphLaunch_v10000",
        "cuLaunchKernel",
        "cudaLaunchKernel_v7000",
        "cudaMemcpyAsync_v3020",
        "cudaStreamSynchronize_v3020",
        "cudaMallocAsync_v11020",
        "cudaFreeAsync_v11020",
        "cudaMemGetInfo_v3020",
    ]
    print("Runtime Calls")
    for name in selected:
        row = fetch_one(
            con,
            f"""
            SELECT COUNT(*) AS n, COALESCE(SUM(r.end - r.start), 0) / 1e6 AS total_ms
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            WHERE s.value = ?
            """,
            (name,),
        )
        per_tok = ""
        if decode_tokens:
            per_tok = f" {row['n'] / decode_tokens:.3f}/tok"
        print(f"  {name:32s} {row['n']:8d}{per_tok:>11s} {row['total_ms']:10.3f} ms")


def print_kernel_summary(con: sqlite3.Connection, top_n: int) -> None:
    total = fetch_one(
        con,
        f"""
        SELECT COUNT(*) AS n,
               COALESCE(SUM(end - start), 0) / 1e6 AS total_ms,
               (MAX(end) - MIN(start)) / 1e6 AS span_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        """,
    )
    print("Kernels")
    print(
        f"  total_count={total['n']} total_kernel_ms={total['total_ms']:.3f} "
        f"kernel_span_ms={total['span_ms']:.3f}"
    )
    for row in con.execute(
        f"""
        SELECT s.value AS name,
               COUNT(*) AS n,
               SUM(k.end - k.start) / 1e6 AS total_ms,
               AVG(k.end - k.start) / 1e3 AS avg_us,
               MAX(k.end - k.start) / 1e3 AS max_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value
        ORDER BY SUM(k.end - k.start) DESC
        LIMIT ?
        """,
        (top_n,),
    ):
        print(
            f"  {row['total_ms']:9.3f} ms {row['n']:6d} "
            f"avg {row['avg_us']:8.3f} us max {row['max_us']:8.3f} us  "
            f"{row['name'][:150]}"
        )


def print_nvtx_summary(con: sqlite3.Connection, top_n: int) -> None:
    if not fetch_one(
        con,
        "SELECT COUNT(*) AS n FROM sqlite_master WHERE type = 'table' AND name = 'NVTX_EVENTS'",
    )["n"]:
        return
    print("NVTX Ranges")
    for row in con.execute(
        f"""
        SELECT text,
               COUNT(*) AS n,
               SUM(end - start) / 1e6 AS total_ms,
               AVG(end - start) / 1e3 AS avg_us,
               MAX(end - start) / 1e6 AS max_ms
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND text IS NOT NULL
        GROUP BY text
        ORDER BY SUM(end - start) DESC
        LIMIT ?
        """,
        (top_n,),
    ):
        print(
            f"  {row['total_ms']:9.3f} ms {row['n']:6d} "
            f"avg {row['avg_us']:9.3f} us max {row['max_ms']:8.3f} ms  {row['text']}"
        )


def print_graph_summary(con: sqlite3.Connection, decode_tokens: int | None) -> None:
    if not fetch_one(
        con,
        "SELECT COUNT(*) AS n FROM sqlite_master WHERE type = 'table' AND name = 'CUDA_GRAPH_EVENTS'",
    )["n"]:
        return
    print("CUDA Graph Events")
    for row in con.execute(
        """
        SELECT s.value AS name, COUNT(*) AS n
        FROM CUDA_GRAPH_EVENTS g
        JOIN StringIds s ON g.nameId = s.id
        GROUP BY s.value
        ORDER BY n DESC
        """
    ):
        per_tok = ""
        if decode_tokens:
            per_tok = f" {row['n'] / decode_tokens:.3f}/tok"
        print(f"  {row['name']:24s} {row['n']:8d}{per_tok:>11s}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument("--decode-tokens", type=int, help="Decode-token count for per-token rates")
    parser.add_argument("--top", type=int, default=20, help="Number of top rows to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = sqlite3.connect(args.sqlite)
    con.row_factory = sqlite3.Row
    print(f"Report: {args.sqlite}")
    print_runtime_summary(con, args.decode_tokens)
    print_graph_summary(con, args.decode_tokens)
    print_kernel_summary(con, args.top)
    print_nvtx_summary(con, args.top)


if __name__ == "__main__":
    main()
