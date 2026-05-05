"""Unit tests for the Nsight decode comparison helper."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_nsys_decode.py"


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_nsys_decode", COMPARE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_trace_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("create table StringIds(id integer primary key, value text)")
    con.execute(
        "create table CUPTI_ACTIVITY_KIND_KERNEL("
        "demangledName integer, start integer, end integer)"
    )
    con.execute(
        "create table CUPTI_ACTIVITY_KIND_RUNTIME(nameId integer, start integer, end integer)"
    )
    con.executemany(
        "insert into StringIds(id, value) values (?, ?)",
        [
            (1, "cublasLt::gemv_decode_kernel"),
            (2, "flash_attention_paged_decode"),
            (3, "softmax_sample_top_p"),
            (4, "cudaGraphLaunch"),
            (5, "cudaLaunchKernel"),
            (6, "cudaMemcpyAsync"),
        ],
    )
    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL(demangledName, start, end) values (?, ?, ?)",
        [
            (1, 0, 200_000),
            (1, 300_000, 500_000),
            (1, 600_000, 800_000),
            (2, 900_000, 950_000),
            (3, 1_000_000, 1_030_000),
        ],
    )
    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME(nameId, start, end) values (?, ?, ?)",
        [
            (4, 0, 40_000),
            (4, 50_000, 90_000),
            (5, 100_000, 120_000),
            (6, 130_000, 140_000),
        ],
    )
    return con


def test_infer_decode_tokens_uses_dominant_decode_gemv(capsys: pytest.CaptureFixture[str]) -> None:
    compare = load_compare_module()
    con = make_trace_db()

    assert compare.infer_decode_tokens(con, "MLC") == 3
    assert "inferred_MLC_tokens=3" in capsys.readouterr().out


def test_infer_decode_tokens_falls_back_to_graph_launches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    compare = load_compare_module()
    con = sqlite3.connect(":memory:")
    con.execute("create table StringIds(id integer primary key, value text)")
    con.execute(
        "create table CUPTI_ACTIVITY_KIND_KERNEL("
        "demangledName integer, start integer, end integer)"
    )
    con.execute(
        "create table CUPTI_ACTIVITY_KIND_RUNTIME(nameId integer, start integer, end integer)"
    )
    con.executemany(
        "insert into StringIds(id, value) values (?, ?)",
        [(1, "fused_mrope_kernel"), (2, "cudaGraphLaunch")],
    )
    con.execute(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL(demangledName, start, end) values (?, ?, ?)",
        (1, 0, 900),
    )
    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME(nameId, start, end) values (?, ?, ?)",
        [(2, 0, 1), (2, 2, 3), (2, 4, 5)],
    )

    assert compare.infer_decode_tokens(con, "MLC") == 3
    assert "from_runtime=cudaGraphLaunch" in capsys.readouterr().out


def test_summarize_kernels_classifies_and_normalizes_per_token() -> None:
    compare = load_compare_module()
    con = make_trace_db()

    summary = compare.summarize_kernels(con, tokens=3)

    assert summary["gemv_gemm"] == pytest.approx((1.0, 200.0))
    assert summary["flash_attention"] == pytest.approx((1 / 3, 50 / 3))
    assert summary["sampling_logits"] == pytest.approx((1 / 3, 10.0))


def test_summarize_runtime_classifies_and_normalizes_per_token() -> None:
    compare = load_compare_module()
    con = make_trace_db()

    summary = compare.summarize_runtime(con, tokens=2)

    assert summary["graph_launch"] == pytest.approx((1.0, 40.0))
    assert summary["kernel_launch"] == pytest.approx((0.5, 10.0))
    assert summary["memcpy"] == pytest.approx((0.5, 5.0))


def test_resolve_tokens_rejects_non_positive_manual_values() -> None:
    compare = load_compare_module()
    con = make_trace_db()

    with pytest.raises(ValueError, match="must be positive"):
        compare.resolve_tokens("0", con, "vLLM")


def test_infer_decode_tokens_fails_without_kernel_table() -> None:
    compare = load_compare_module()
    con = sqlite3.connect(":memory:")

    with pytest.raises(ValueError, match="no kernel table"):
        compare.infer_decode_tokens(con, "MLC")
