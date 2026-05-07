"""Unit tests for the Qwen3.5 MLC/vLLM benchmark wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
import base64
import subprocess
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = REPO_ROOT / "benchmark_qwen35_vllm_mlc.py"
VLLM_SCRIPT = REPO_ROOT / "run_vllm_im.py"
TRANSFORMERS_SCRIPT = REPO_ROOT / "run_transformers_im.py"
MLC_RUNNER_SCRIPT = REPO_ROOT / "run_im.py"
PARITY_SCRIPT = REPO_ROOT / "check_qwen35_im_parity.py"
PREPARE_VLLM_SCRIPT = REPO_ROOT / "prepare_vllm_qwen35_im_model.py"
CUDA_GRAPH_ANALYZER_SCRIPT = REPO_ROOT / "analyze_cuda_graph_dump.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_benchmark_module():
    return load_module("qwen35_benchmark", BENCHMARK_SCRIPT)


def load_mlc_runner_module():
    return load_module("qwen35_mlc_runner", MLC_RUNNER_SCRIPT)


def load_parity_module():
    return load_module("qwen35_parity_runner", PARITY_SCRIPT)


def load_prepare_vllm_module():
    return load_module("qwen35_prepare_vllm", PREPARE_VLLM_SCRIPT)


def load_cuda_graph_analyzer_module():
    return load_module("qwen35_cuda_graph_analyzer", CUDA_GRAPH_ANALYZER_SCRIPT)


def load_vllm_module():
    return load_module("qwen35_vllm_runner", VLLM_SCRIPT)


def load_transformers_module():
    return load_module("qwen35_transformers_runner", TRANSFORMERS_SCRIPT)


def make_args(**overrides):
    args = {
        "image": "kentucky.png",
        "prompt": "What is in the image?",
        "fit_image_width": 512,
        "fit_image_height": 512,
        "max_tokens": 128,
        "warmup_runs": 1,
        "benchmark_runs": 5,
        "processor_use_fast": "auto",
        "hf_model": "familiar-ai/logos-multitask-qwen3.5-2026-05-03-best",
        "mlc_model": "dist/example-MLC",
        "mlc_model_lib": "dist/libs/example-cuda.so",
        "mlc_defer_cpu_token_burst": "auto",
        "mlc_decode_burst_steps": 16,
        "mlc_sync_decode_timing": "auto",
        "mlc_kv_cache_page_size": None,
        "keep_eos": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_parse_image_grid_normalizes_mlc_and_vllm_output() -> None:
    benchmark = load_benchmark_module()

    assert benchmark.parse_image_grid("image_grid=(1, 32, 32) image_embed=256") == "1x32x32"
    assert (
        benchmark.parse_image_grid("image_grid_thw=tensor([[ 1, 32, 32]])")
        == "1x32x32"
    )
    assert benchmark.parse_image_grid("pixel_values_shape=(1024, 1536)") is None


def test_print_summary_rejects_image_grid_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark = load_benchmark_module()
    rows = [
        (
            "MLC",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_second": "404.0",
                "generated_tokens_per_run": "128",
            },
        ),
        (
            "vLLM",
            {
                "image_grid": "1x16x16",
                "generated_tokens_per_second": "408.0",
                "generated_tokens_per_run": "128",
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="different image grids"):
        benchmark.print_summary(
            rows,
            allow_image_grid_mismatch=False,
            allow_token_count_mismatch=False,
            comparable_benchmark=True,
        )

    out = capsys.readouterr().out
    assert "image_grid_mismatch mlc=1x32x32 vllm=1x16x16" in out


def test_print_summary_allows_intentional_grid_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark_module()
    rows = [
        (
            "MLC",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "404.0",
            },
        ),
        (
            "vLLM",
            {
                "image_grid": "1x16x16",
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "408.0",
            },
        ),
    ]

    benchmark.print_summary(
        rows,
        allow_image_grid_mismatch=True,
        allow_token_count_mismatch=False,
        comparable_benchmark=True,
    )

    out = capsys.readouterr().out
    assert "image_grid_mismatch mlc=1x32x32 vllm=1x16x16" in out
    assert "delta_vllm_minus_mlc_tokens_per_second=4.000" in out


def test_print_summary_skips_delta_for_short_or_cold_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark_module()
    rows = [
        ("MLC", {"image_grid": "1x32x32", "generated_tokens_per_second": "404.0"}),
        ("vLLM", {"image_grid": "1x32x32", "generated_tokens_per_second": "408.0"}),
    ]

    benchmark.print_summary(
        rows,
        allow_image_grid_mismatch=False,
        allow_token_count_mismatch=False,
        comparable_benchmark=False,
    )

    out = capsys.readouterr().out
    assert "image_grid_match=1x32x32" in out
    assert "generated_token_count_match=128" not in out
    assert "comparison_warning=short_or_cold_run" in out
    assert "delta_vllm_minus_mlc_tokens_per_second" not in out


def test_print_summary_rejects_generated_token_count_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark_module()
    rows = [
        (
            "MLC",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "404.0",
            },
        ),
        (
            "vLLM",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "96",
                "generated_tokens_per_second": "408.0",
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="different generated token counts"):
        benchmark.print_summary(
            rows,
            allow_image_grid_mismatch=False,
            allow_token_count_mismatch=False,
            comparable_benchmark=True,
        )

    out = capsys.readouterr().out
    assert "generated_token_count_mismatch mlc=128 vllm=96" in out


def test_print_summary_allows_intentional_generated_token_count_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark_module()
    rows = [
        (
            "MLC",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "404.0",
            },
        ),
        (
            "vLLM",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "96",
                "generated_tokens_per_second": "408.0",
            },
        ),
    ]

    benchmark.print_summary(
        rows,
        allow_image_grid_mismatch=False,
        allow_token_count_mismatch=True,
        comparable_benchmark=True,
    )

    out = capsys.readouterr().out
    assert "generated_token_count_mismatch mlc=128 vllm=96" in out
    assert "delta_vllm_minus_mlc_tokens_per_second=4.000" in out


def test_print_summary_rejects_missing_image_grid_for_comparable_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark_module()
    rows = [
        (
            "MLC",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "404.0",
            },
        ),
        (
            "vLLM",
            {
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "408.0",
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="must both report image grids"):
        benchmark.print_summary(
            rows,
            allow_image_grid_mismatch=False,
            allow_token_count_mismatch=False,
            comparable_benchmark=True,
        )

    out = capsys.readouterr().out
    assert "image_grid_missing mlc=1x32x32 vllm=" in out


def test_print_summary_rejects_missing_token_count_for_comparable_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark_module()
    rows = [
        (
            "MLC",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_run": "128",
                "generated_tokens_per_second": "404.0",
            },
        ),
        (
            "vLLM",
            {
                "image_grid": "1x32x32",
                "generated_tokens_per_second": "408.0",
            },
        ),
    ]

    with pytest.raises(RuntimeError, match="must both report generated token counts"):
        benchmark.print_summary(
            rows,
            allow_image_grid_mismatch=False,
            allow_token_count_mismatch=False,
            comparable_benchmark=True,
        )

    out = capsys.readouterr().out
    assert "generated_token_count_missing mlc=128 vllm=" in out


def test_is_comparable_benchmark_thresholds() -> None:
    benchmark = load_benchmark_module()

    assert benchmark.is_comparable_benchmark(
        SimpleNamespace(max_tokens=128, warmup_runs=1, benchmark_runs=5)
    )
    assert not benchmark.is_comparable_benchmark(
        SimpleNamespace(max_tokens=127, warmup_runs=1, benchmark_runs=5)
    )
    assert not benchmark.is_comparable_benchmark(
        SimpleNamespace(max_tokens=128, warmup_runs=0, benchmark_runs=5)
    )
    assert not benchmark.is_comparable_benchmark(
        SimpleNamespace(max_tokens=128, warmup_runs=1, benchmark_runs=4)
    )


def test_benchmark_wrapper_processor_speed_argument_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "false")
    monkeypatch.setattr("sys.argv", ["benchmark_qwen35_vllm_mlc.py", "--image", "kentucky.png"])

    assert benchmark.parse_args().processor_use_fast == "false"


def test_benchmark_wrapper_processor_speed_cli_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "false")
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_qwen35_vllm_mlc.py",
            "--image",
            "kentucky.png",
            "--processor-use-fast",
            "true",
        ],
    )

    assert benchmark.parse_args().processor_use_fast == "true"


def test_benchmark_wrapper_rejects_invalid_processor_speed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "maybe")
    monkeypatch.setattr("sys.argv", ["benchmark_qwen35_vllm_mlc.py", "--image", "kentucky.png"])

    with pytest.raises(ValueError, match="VLM_PROCESSOR_USE_FAST must be one of"):
        benchmark.parse_args()


def test_vllm_command_preserves_remote_model_ids() -> None:
    benchmark = load_benchmark_module()

    command, _ = benchmark.vllm_command(make_args())

    assert "--hf-model" in command
    model_index = command.index("--hf-model") + 1
    assert command[model_index] == "familiar-ai/logos-multitask-qwen3.5-2026-05-03-best"
    processor_index = command.index("--processor-use-fast") + 1
    assert command[processor_index] == "auto"
    assert "--ignore-eos" in command


def test_vllm_command_forwards_explicit_processor_speed() -> None:
    benchmark = load_benchmark_module()

    command, _ = benchmark.vllm_command(make_args(processor_use_fast="false"))

    processor_index = command.index("--processor-use-fast") + 1
    assert command[processor_index] == "false"


def test_vllm_command_forwards_processor_speed_without_hf_model() -> None:
    benchmark = load_benchmark_module()

    command, _ = benchmark.vllm_command(
        make_args(hf_model=None, processor_use_fast="true")
    )

    assert "--hf-model" not in command
    processor_index = command.index("--processor-use-fast") + 1
    assert command[processor_index] == "true"


def test_vllm_command_resolves_existing_local_model_paths() -> None:
    benchmark = load_benchmark_module()

    command, _ = benchmark.vllm_command(make_args(hf_model="benchmark_qwen35_vllm_mlc.py"))

    assert "--hf-model" in command
    model_index = command.index("--hf-model") + 1
    assert command[model_index] == str(BENCHMARK_SCRIPT.resolve())


def test_fixed_canvas_image_resize_matches_across_runners(tmp_path: Path) -> None:
    from PIL import Image

    mlc_runner = load_mlc_runner_module()
    vllm_runner = load_vllm_module()
    transformers_runner = load_transformers_module()
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 2), (10, 20, 30)).save(source)

    vllm_image = vllm_runner.load_image(str(source), 8, 8)
    transformers_image = transformers_runner.load_image(
        str(source),
        fit_size=0,
        square_canvas=False,
        fit_width=8,
        fit_height=8,
    )
    mlc_data_url = mlc_runner.image_file_to_url(str(source), fit_width=8, fit_height=8)
    _, encoded = mlc_data_url.split(",", 1)
    mlc_image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    assert vllm_image.size == (8, 8)
    assert transformers_image.size == (8, 8)
    assert mlc_image.size == (8, 8)
    assert vllm_image.getpixel((0, 0)) == (255, 255, 255)
    assert transformers_image.getpixel((0, 0)) == (255, 255, 255)
    assert mlc_image.getpixel((0, 0)) == (255, 255, 255)


def test_compile_im_dry_run_preserves_semicolon_arguments() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OPT": "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1",
            "PREFILL_CHUNK_SIZE": "320",
            "CONTEXT_WINDOW_SIZE": "2048",
        }
    )

    result = subprocess.run(
        [
            str(REPO_ROOT / "compile_im.sh"),
            "familiar-ai/logos-multitask-qwen3.5-2026-05-03-best",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0
    assert "dry_run " in result.stdout
    assert "explicitcrosscg-appendmetacg-relaxedmeta-outputcapture-noverify-pc320-cuda.so" in result.stdout
    assert "--opt flashinfer=1\\;cublas_gemm=1\\;cudagraph=1\\;cutlass=1" in result.stdout
    assert (
        "--overrides context_window_size=2048\\;prefill_chunk_size=320\\;max_batch_size=1"
        in result.stdout
    )
    assert "command not found" not in result.stdout


def test_compile_im_defaults_to_current_best_graph_capture_knobs() -> None:
    script = (REPO_ROOT / "compile_im.sh").read_text(encoding="utf-8")

    assert 'MLC_QWEN35_EXPLICIT_PAGED_CROSS_ATTENTION:-1' in script
    assert 'TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_CROSS_ATTENTION:-1' in script
    assert 'TVM_CUDA_GRAPH_CAPTURE_KV_CACHE_APPEND_METADATA:-1' in script
    assert 'TVM_CUDA_GRAPH_RELAXED_KV_CACHE_METADATA_TUPLES:-1' in script
    assert 'TVM_CUDA_GRAPH_CAPTURE_FUNC_OUTPUTS:-1' in script
    assert 'MLC_QWEN35_OMIT_VERIFY="${MLC_QWEN35_OMIT_VERIFY:-1}"' in script
    assert 'DEFAULT_LIB_SUFFIX="${DEFAULT_LIB_SUFFIX}-noverify-pc${PREFILL_CHUNK_SIZE}"' in script
    assert 'LIB_SUFFIX="${LIB_SUFFIX:-$DEFAULT_LIB_SUFFIX}"' in script


def test_cuda_graph_dump_analyzer_can_gate_current_decode_shape(tmp_path: Path) -> None:
    analyzer = load_cuda_graph_analyzer_module()
    dump = tmp_path / "debug-phase6.py"
    dump.write_text(
        dedent(
            '''
            # from tvm.script import relax as R
            class Module:
                @R.function
                def decode_mrope(x):
                    y = R.call_packed("vm.builtin.attention_kv_cache_get_query_positions", x)
                    z = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (x,))
                    w = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (z,))
                    return w
            '''
        ),
        encoding="utf-8",
    )

    stats = analyzer.inspect_function_stats(dump, "decode_mrope")

    assert stats["cuda_graph_calls"] == 2
    assert stats["packed"]["vm.builtin.attention_kv_cache_get_query_positions"] == 1
    analyzer.check_function_expectations(
        stats,
        expect_cuda_graph_calls=2,
        expect_packed_call=["vm.builtin.attention_kv_cache_get_query_positions=1"],
        expect_no_packed_call=["vm.builtin.attention_kv_cache_append_mha_kv"],
    )
    with pytest.raises(SystemExit, match="Expected 7 CUDA graph calls"):
        analyzer.check_function_expectations(
            stats,
            expect_cuda_graph_calls=7,
            expect_packed_call=[],
            expect_no_packed_call=[],
        )


def test_cuda_graph_dump_analyzer_filters_exact_capture_parent(tmp_path: Path) -> None:
    analyzer = load_cuda_graph_analyzer_module()
    dump = tmp_path / "debug-phase6.py"
    dump.write_text(
        dedent(
            '''
            # from tvm.script import relax as R
            class Module:
                @R.function(private=True)
                def decode_cuda_graph_capture(x):
                    return x

                @R.function(private=True)
                def decode_cuda_graph_capture1(x):
                    return x

                @R.function(private=True)
                def batch_decode_cuda_graph_capture(x):
                    return x
            '''
        ),
        encoding="utf-8",
    )

    funcs = {
        "decode_cuda_graph_capture": {},
        "decode_cuda_graph_capture1": {},
        "batch_decode_cuda_graph_capture": {},
    }

    assert analyzer.capture_parent_name("decode_cuda_graph_capture1") == "decode"
    assert analyzer.capture_parent_name("batch_decode_cuda_graph_capture") == "batch_decode"

    selected = [
        name
        for name in funcs
        if analyzer.capture_parent_name(name) == "decode"
    ]
    assert selected == ["decode_cuda_graph_capture", "decode_cuda_graph_capture1"]


def test_cuda_graph_dump_analyzer_summarizes_ordered_capture_regions(
    tmp_path: Path,
) -> None:
    analyzer = load_cuda_graph_analyzer_module()
    dump = tmp_path / "debug-phase6.py"
    dump.write_text(
        dedent(
            '''
            # from tvm.script import relax as R
            class Module:
                @R.function(private=True)
                def decode_cuda_graph_capture(x):
                    y = R.call_packed("vm.builtin.attention_kv_cache_append_mha_kv", x)
                    z = cls.fused_linear(x)
                    return z

                @R.function(private=True)
                def decode_cuda_graph_capture2(x):
                    y = R.call_builtin_with_ctx("vm.builtin.attention_kv_cache_cross_attention_with_paged_metadata", (x,))
                    z = R.call_dps_packed("vm.builtin.attention_kv_cache_get_paged_decode_metadata", x)
                    return z

                @R.function(private=True)
                def decode_cuda_graph_capture1(x):
                    y = R.call_packed("vm.builtin.attention_kv_cache_cross_attention_with_paged_metadata", x)
                    z = cls.fused_mlp(x)
                    return z
            '''
        ),
        encoding="utf-8",
    )

    regions = analyzer.get_capture_region_summaries(
        dump,
        parent_filter="decode",
        max_events=2,
    )

    assert [region["name"] for region in regions] == [
        "decode_cuda_graph_capture",
        "decode_cuda_graph_capture1",
        "decode_cuda_graph_capture2",
    ]
    assert [region["index"] for region in regions] == [0, 1, 2]
    assert analyzer.capture_index("decode_cuda_graph_capture") == 0
    assert analyzer.capture_index("decode_cuda_graph_capture2") == 2
    assert regions[0]["packed"]["vm.builtin.attention_kv_cache_append_mha_kv"] == 1
    assert regions[1]["tir"]["fused_mlp"] == 1
    assert (
        regions[2]["builtin"][
            "vm.builtin.attention_kv_cache_cross_attention_with_paged_metadata"
        ]
        == 1
    )
    assert (
        regions[2]["packed"][
            "vm.builtin.attention_kv_cache_get_paged_decode_metadata"
        ]
        == 1
    )
    assert regions[0]["first_events"][0][1:] == (
        "packed",
        "vm.builtin.attention_kv_cache_append_mha_kv",
    )


def test_mlc_command_sets_current_runtime_defaults() -> None:
    benchmark = load_benchmark_module()

    command, env = benchmark.mlc_command(make_args())

    assert command[0] == str(REPO_ROOT / "run_im.sh")
    assert env["MLC_MODEL"] == str((REPO_ROOT / "dist/example-MLC").resolve())
    assert env["MLC_MODEL_LIB"] == str((REPO_ROOT / "dist/libs/example-cuda.so").resolve())
    assert env["MLC_DEFER_CPU_TOKEN_BURST"] == "1"
    assert env["MLC_DECODE_BURST_STEPS"] == "16"
    assert env["MLC_STABLE_DECODE_EMBEDDING"] == "1"
    assert "MLC_SYNC_DECODE_TIMING" not in env
    assert "--ignore-eos" in command


def test_mlc_command_honors_runtime_overrides() -> None:
    benchmark = load_benchmark_module()

    command, env = benchmark.mlc_command(
        make_args(
            mlc_defer_cpu_token_burst="0",
            mlc_decode_burst_steps=8,
            mlc_sync_decode_timing="1",
            mlc_kv_cache_page_size=32,
        )
    )

    assert env["MLC_DEFER_CPU_TOKEN_BURST"] == "0"
    assert env["MLC_DECODE_BURST_STEPS"] == "8"
    assert env["MLC_SYNC_DECODE_TIMING"] == "1"
    assert env["MLC_KV_CACHE_PAGE_SIZE"] == "32"
    assert env["MLC_ALLOW_EXPERIMENTAL_KV_CACHE_PAGE_SIZE"] == "1"
    page_size_index = command.index("--kv-cache-page-size")
    assert command[page_size_index + 1] == "32"


def test_run_command_parses_mlc_metrics_without_launching_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "image_grid=(1, 32, 32) image_embed=256\n"
                "mlc_engine_metrics decode_tokens_per_second=433.19 "
                "decode_seconds=1.23\n"
                "mlc_effective_decode_metrics generated_tokens_per_run=128 "
                "effective_decode_tokens_per_second=435.04\n"
                "benchmark warmup_runs=1 measured_runs=5 generated_tokens_per_run=128 "
                "engine_init_seconds=0.42 avg_generate_seconds=0.316 "
                "generated_tokens_per_second=404.70\n"
            ),
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    name, parsed = benchmark.run_command(
        "MLC",
        ["run_im.sh"],
        {
            "MLC_DEFER_CPU_TOKEN_BURST": "1",
            "MLC_DECODE_BURST_STEPS": "16",
            "MLC_STABLE_DECODE_EMBEDDING": "1",
            "MLC_KV_CACHE_PAGE_SIZE": "16",
            "MLC_SYNC_DECODE_TIMING": "1",
        },
    )

    assert name == "MLC"
    assert parsed["image_grid"] == "1x32x32"
    assert parsed["generated_tokens_per_second"] == "404.70"
    assert parsed["mlc_decode_tokens_per_second"] == "433.19"
    assert parsed["mlc_effective_decode_tokens_per_second"] == "435.04"
    assert parsed["mlc_defer_cpu_token_burst"] == "1"
    assert parsed["mlc_sync_decode_timing"] == "1"


def test_run_command_parses_vllm_metrics_without_launching_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "image_grid_thw=tensor([[ 1, 32, 32]])\n"
                "vllm_cache_config block_size=16 user_specified_block_size=None "
                "mamba_block_size=16 mamba_cache_mode=none mamba_page_size_padded=256\n"
                "benchmark warmup_runs=1 measured_runs=5 generated_tokens_per_run=128 "
                "model_load_seconds=26.78 avg_generate_seconds=0.313 "
                "generated_tokens_per_second=408.63\n"
            ),
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    name, parsed = benchmark.run_command("vLLM", ["run_vllm_im.sh"], {})

    assert name == "vLLM"
    assert parsed["image_grid"] == "1x32x32"
    assert parsed["generated_tokens_per_second"] == "408.63"
    assert parsed["model_load_seconds"] == "26.78"
    assert parsed["vllm_block_size"] == "16"


def test_run_command_raises_on_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=2, stdout="usage error\n")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="MLC command failed with exit code 2"):
        benchmark.run_command("MLC", ["run_im.sh"], {})


def test_run_command_raises_when_benchmark_line_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="image_grid=(1, 32, 32)\n")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Could not parse benchmark line for vLLM"):
        benchmark.run_command("vLLM", ["run_vllm_im.sh"], {})


def test_vllm_runner_default_model_path_prefers_environment_override() -> None:
    vllm_runner = load_vllm_module()
    previous = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = "override/model"
    try:
        assert vllm_runner.default_model_path() == "override/model"
    finally:
        if previous is None:
            os.environ.pop("HF_MODEL", None)
        else:
            os.environ["HF_MODEL"] = previous


def test_vllm_runner_default_model_path_uses_local_copy_when_present() -> None:
    vllm_runner = load_vllm_module()
    previous = os.environ.pop("HF_MODEL", None)
    try:
        expected = (
            str(vllm_runner.DEFAULT_LOCAL_VLLM_MODEL)
            if vllm_runner.DEFAULT_LOCAL_VLLM_MODEL.exists()
            else vllm_runner.DEFAULT_HF_MODEL
        )
        assert vllm_runner.default_model_path() == expected
    finally:
        if previous is not None:
            os.environ["HF_MODEL"] = previous


def test_vllm_runner_processor_speed_argument_defaults_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_runner = load_vllm_module()
    monkeypatch.delenv("VLM_PROCESSOR_USE_FAST", raising=False)
    monkeypatch.setattr("sys.argv", ["run_vllm_im.py", "--image", "kentucky.png"])

    assert vllm_runner.parse_args().processor_use_fast == "auto"


def test_vllm_runner_rejects_invalid_processor_speed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_runner = load_vllm_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "maybe")
    monkeypatch.setattr("sys.argv", ["run_vllm_im.py", "--image", "kentucky.png"])

    with pytest.raises(ValueError, match="VLM_PROCESSOR_USE_FAST must be one of"):
        vllm_runner.parse_args()


def test_transformers_runner_processor_speed_argument_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers_runner = load_transformers_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "false")
    monkeypatch.setattr("sys.argv", ["run_transformers_im.py", "--image", "kentucky.png"])

    assert transformers_runner.parse_args().processor_use_fast == "false"


def test_transformers_runner_exposes_ignore_eos_for_fixed_token_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers_runner = load_transformers_module()
    monkeypatch.delenv("VLM_PROCESSOR_USE_FAST", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["run_transformers_im.py", "--image", "kentucky.png", "--ignore-eos"],
    )

    assert transformers_runner.parse_args().ignore_eos


def test_transformers_runner_rejects_invalid_processor_speed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers_runner = load_transformers_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "maybe")
    monkeypatch.setattr("sys.argv", ["run_transformers_im.py", "--image", "kentucky.png"])

    with pytest.raises(ValueError, match="VLM_PROCESSOR_USE_FAST must be one of"):
        transformers_runner.parse_args()


def test_parity_helper_processor_speed_argument_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parity = load_parity_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "true")
    monkeypatch.setattr("sys.argv", ["check_qwen35_im_parity.py", "--image", "kentucky.png"])

    assert parity.parse_args().processor_use_fast == "true"


def test_parity_helper_rejects_invalid_processor_speed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parity = load_parity_module()
    monkeypatch.setenv("VLM_PROCESSOR_USE_FAST", "maybe")
    monkeypatch.setattr("sys.argv", ["check_qwen35_im_parity.py", "--image", "kentucky.png"])

    with pytest.raises(ValueError, match="VLM_PROCESSOR_USE_FAST must be one of"):
        parity.parse_args()


def test_prepare_vllm_qwen35_key_mapping() -> None:
    prepare_vllm = load_prepare_vllm_module()

    assert (
        prepare_vllm.map_key(
            "model.language_model.language_model.language_model.layers.0.input_layernorm.weight"
        )
        == "language_model.model.layers.0.input_layernorm.weight"
    )
    assert (
        prepare_vllm.map_key("model.language_model.visual.patch_embed.proj.weight")
        == "visual.patch_embed.proj.weight"
    )
    assert prepare_vllm.map_key("unrelated.weight") == "unrelated.weight"


def test_prepare_vllm_qwen35_copy_config_files_skips_weights(tmp_path: Path) -> None:
    prepare_vllm = load_prepare_vllm_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text("{}\n")
    (source / "tokenizer.json").write_text("{}\n")
    (source / "model.safetensors").write_text("skip\n")
    (source / "model.safetensors.index.json").write_text("skip\n")
    (source / ".gitattributes").write_text("skip\n")

    prepare_vllm.copy_config_files(source, output)

    assert sorted(path.name for path in output.iterdir()) == ["config.json", "tokenizer.json"]


def test_prepare_tokenizer_path_respects_explicit_tokenizer() -> None:
    vllm_runner = load_vllm_module()

    assert vllm_runner.prepare_tokenizer_path("unused/model", "explicit-tokenizer") == (
        "explicit-tokenizer"
    )


def test_prepare_tokenizer_path_returns_local_source_for_standard_tokenizer(
    tmp_path: Path,
) -> None:
    vllm_runner = load_vllm_module()
    source = tmp_path / "model"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2TokenizerFast"}) + "\n"
    )

    assert vllm_runner.prepare_tokenizer_path(str(source), None) == str(source)


def test_prepare_tokenizer_path_rewrites_tokenizers_backend_for_vllm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_runner = load_vllm_module()
    source = tmp_path / "model"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "extra": True}) + "\n"
    )
    (source / "tokenizer.json").write_text("{}\n")
    monkeypatch.chdir(tmp_path)

    output = Path(vllm_runner.prepare_tokenizer_path(str(source), None))
    output_dir = tmp_path / output

    assert output == Path("dist") / "vllm-tokenizers" / "model-qwen2"
    assert (output_dir / "tokenizer.json").read_text() == "{}\n"
    rewritten_config = json.loads((output_dir / "tokenizer_config.json").read_text())
    assert rewritten_config == {"tokenizer_class": "Qwen2TokenizerFast", "extra": True}
