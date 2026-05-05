#!/usr/bin/env python3
"""Run apples-to-apples Qwen3.5 image benchmarks for vLLM and MLC."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MLC_MODEL = (
    "dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC"
)
DEFAULT_MLC_MODEL_LIB = (
    "dist/libs/"
    "logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-"
    "explicit-paged-stabledecodeinput-cuda.so"
)
DEFAULT_VLLM_MODEL = "dist/logos-multitask-qwen3.5-2026-05-03-best-vllm-hf-tokenizercompat"
PROCESSOR_USE_FAST_CHOICES = ("auto", "true", "false")


def resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def processor_use_fast_default() -> str:
    value = os.environ.get("VLM_PROCESSOR_USE_FAST", "auto")
    if value not in PROCESSOR_USE_FAST_CHOICES:
        raise ValueError(
            "VLM_PROCESSOR_USE_FAST must be one of: "
            + ", ".join(PROCESSOR_USE_FAST_CHOICES)
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="kentucky.png")
    parser.add_argument("--prompt", default="What is in the image?")
    parser.add_argument("--fit-image-width", type=int, default=512)
    parser.add_argument("--fit-image-height", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--processor-use-fast",
        choices=PROCESSOR_USE_FAST_CHOICES,
        default=processor_use_fast_default(),
        help="Pass use_fast to vLLM/Transformers AutoProcessor paths. 'auto' preserves default.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=5,
        help="Measured runs. Values below 2 are promoted to 2 so MLC emits benchmark metrics.",
    )
    default_vllm_model = DEFAULT_VLLM_MODEL if (REPO_ROOT / DEFAULT_VLLM_MODEL).exists() else None
    parser.add_argument("--hf-model", default=os.environ.get("HF_MODEL", default_vllm_model))
    parser.add_argument("--mlc-model", default=os.environ.get("MLC_MODEL", DEFAULT_MLC_MODEL))
    parser.add_argument(
        "--mlc-model-lib",
        default=os.environ.get("MLC_MODEL_LIB", DEFAULT_MLC_MODEL_LIB),
    )
    parser.add_argument(
        "--mlc-defer-cpu-token-burst",
        choices=("auto", "0", "1"),
        default="auto",
        help="MLC_DEFER_CPU_TOKEN_BURST for MLC runs. 'auto' uses the benchmark default.",
    )
    parser.add_argument(
        "--mlc-decode-burst-steps",
        type=int,
        default=16,
        help="MLC_DECODE_BURST_STEPS for MLC runs.",
    )
    parser.add_argument(
        "--mlc-sync-decode-timing",
        choices=("auto", "0", "1"),
        default="auto",
        help=(
            "MLC_SYNC_DECODE_TIMING for MLC runs. Use 1 for synchronized phase attribution; "
            "auto preserves the environment/default fast timing."
        ),
    )
    parser.add_argument(
        "--mlc-kv-cache-page-size",
        type=int,
        default=None,
        help="MLC KV cache page size. Requires a model lib compiled with the same page size.",
    )
    parser.add_argument(
        "--vllm-first",
        action="store_true",
        help="Run vLLM before MLC. Default is MLC then vLLM.",
    )
    parser.add_argument(
        "--keep-eos",
        action="store_true",
        help="Do not pass --ignore-eos. Fixed-token benchmarking should leave this unset.",
    )
    parser.add_argument(
        "--engine",
        choices=("both", "mlc", "vllm"),
        default="both",
        help="Run both engines, or only one side for smoke testing.",
    )
    parser.add_argument(
        "--allow-image-grid-mismatch",
        action="store_true",
        help="Do not fail when both engines report different image grids.",
    )
    parser.add_argument(
        "--allow-token-count-mismatch",
        action="store_true",
        help="Do not fail when both engines report different generated token counts.",
    )
    return parser.parse_args()


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in re.finditer(r"([a-zA-Z_]+)=([^ ]+)", line):
        values[match.group(1)] = match.group(2)
    return values


def parse_image_grid(line: str) -> str | None:
    mlc_match = re.search(r"image_grid=\((\d+),\s*(\d+),\s*(\d+)\)", line)
    if mlc_match:
        return "x".join(mlc_match.groups())
    vllm_match = re.search(r"image_grid_thw=tensor\(\[\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]\]\)", line)
    if vllm_match:
        return "x".join(vllm_match.groups())
    return None


def run_command(name: str, command: list[str], env: dict[str, str]) -> tuple[str, dict[str, str]]:
    print(f"\n## {name}")
    print("command=" + " ".join(shlex.quote(part) for part in command))
    if name == "MLC":
        print(
            "mlc_runtime_config "
            f"defer_cpu_token_burst={env.get('MLC_DEFER_CPU_TOKEN_BURST', '')} "
            f"decode_burst_steps={env.get('MLC_DECODE_BURST_STEPS', '')} "
            f"stable_decode_embedding={env.get('MLC_STABLE_DECODE_EMBEDDING', '')} "
            f"kv_cache_page_size={env.get('MLC_KV_CACHE_PAGE_SIZE', '16')} "
            f"sync_decode_timing={env.get('MLC_SYNC_DECODE_TIMING', '')}"
        )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"{name} command failed with exit code {result.returncode}")
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("benchmark "):
            parsed.update(parse_key_values(line))
        elif line.startswith("mlc_engine_metrics "):
            for key, value in parse_key_values(line).items():
                parsed[f"mlc_{key}"] = value
        elif line.startswith("mlc_effective_decode_metrics "):
            for key, value in parse_key_values(line).items():
                if key.startswith("effective_"):
                    parsed[f"mlc_{key}"] = value
                else:
                    parsed[f"mlc_effective_{key}"] = value
        elif line.startswith("mlc_runtime_config "):
            for key, value in parse_key_values(line).items():
                parsed[f"mlc_{key}"] = value
        elif line.startswith("image_grid"):
            image_grid = parse_image_grid(line)
            if image_grid:
                parsed["image_grid"] = image_grid
            for key, value in parse_key_values(line).items():
                parsed[f"image_{key}"] = value
        elif line.startswith("vllm_cache_config "):
            for key, value in parse_key_values(line).items():
                parsed[f"vllm_{key}"] = value
    if name == "MLC":
        parsed["mlc_defer_cpu_token_burst"] = env.get("MLC_DEFER_CPU_TOKEN_BURST", "")
        parsed["mlc_decode_burst_steps"] = env.get("MLC_DECODE_BURST_STEPS", "")
        parsed["mlc_stable_decode_embedding"] = env.get("MLC_STABLE_DECODE_EMBEDDING", "")
        parsed["mlc_kv_cache_page_size"] = env.get("MLC_KV_CACHE_PAGE_SIZE", "16")
        parsed["mlc_sync_decode_timing"] = env.get("MLC_SYNC_DECODE_TIMING", "")
    if "generated_tokens_per_second" not in parsed:
        raise RuntimeError(f"Could not parse benchmark line for {name}")
    return name, parsed


def mlc_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["MLC_MODEL"] = resolve_repo_path(args.mlc_model)
    env["MLC_MODEL_LIB"] = resolve_repo_path(args.mlc_model_lib)
    if args.mlc_defer_cpu_token_burst == "auto":
        env.setdefault("MLC_DEFER_CPU_TOKEN_BURST", "1")
    else:
        env["MLC_DEFER_CPU_TOKEN_BURST"] = args.mlc_defer_cpu_token_burst
    env["MLC_DECODE_BURST_STEPS"] = str(args.mlc_decode_burst_steps)
    env.setdefault("MLC_STABLE_DECODE_EMBEDDING", "1")
    if args.mlc_sync_decode_timing != "auto":
        env["MLC_SYNC_DECODE_TIMING"] = args.mlc_sync_decode_timing
    if args.mlc_kv_cache_page_size is not None:
        env["MLC_KV_CACHE_PAGE_SIZE"] = str(args.mlc_kv_cache_page_size)
        if args.mlc_kv_cache_page_size != 16:
            env.setdefault("MLC_ALLOW_EXPERIMENTAL_KV_CACHE_PAGE_SIZE", "1")
    command = [
        str(REPO_ROOT / "run_im.sh"),
        "--image",
        args.image,
        "--fit-image-width",
        str(args.fit_image_width),
        "--fit-image-height",
        str(args.fit_image_height),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--benchmark-runs",
        str(args.benchmark_runs),
    ]
    if args.mlc_kv_cache_page_size is not None:
        command.extend(["--kv-cache-page-size", str(args.mlc_kv_cache_page_size)])
    if not args.keep_eos:
        command.append("--ignore-eos")
    return command, env


def vllm_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    command = [
        str(REPO_ROOT / "run_vllm_im.sh"),
        "--image",
        args.image,
        "--fit-image-width",
        str(args.fit_image_width),
        "--fit-image-height",
        str(args.fit_image_height),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--benchmark-runs",
        str(args.benchmark_runs),
    ]
    if args.hf_model:
        hf_model_path = Path(args.hf_model).expanduser()
        hf_model = resolve_repo_path(args.hf_model) if hf_model_path.exists() else args.hf_model
        command.extend(["--hf-model", hf_model])
    command.extend(["--processor-use-fast", args.processor_use_fast])
    if not args.keep_eos:
        command.append("--ignore-eos")
    return command, env


def is_comparable_benchmark(args: argparse.Namespace) -> bool:
    return args.max_tokens >= 128 and args.warmup_runs >= 1 and args.benchmark_runs >= 5


def print_summary(
    rows: list[tuple[str, dict[str, str]]],
    allow_image_grid_mismatch: bool,
    allow_token_count_mismatch: bool,
    comparable_benchmark: bool,
) -> None:
    print("\n## summary")
    print(
        "engine\timage_grid\tstartup_seconds\tmlc_runtime\tgenerated_tokens_per_run\t"
        "avg_generate_seconds\tgenerated_tokens_per_second\t"
        "mlc_effective_decode_tokens_per_second"
    )
    for name, data in rows:
        startup_seconds = data.get("engine_init_seconds", data.get("model_load_seconds", ""))
        mlc_runtime = ""
        if name == "MLC":
            mlc_runtime = (
                f"defer={data.get('mlc_defer_cpu_token_burst', '')},"
                f"burst={data.get('mlc_decode_burst_steps', '')},"
                f"stable={data.get('mlc_stable_decode_embedding', '')},"
                f"page={data.get('mlc_kv_cache_page_size', '')},"
                f"sync={data.get('mlc_sync_decode_timing', '')}"
            )
        print(
            "\t".join(
                [
                    name,
                    data.get("image_grid", ""),
                    startup_seconds,
                    mlc_runtime,
                    data.get("generated_tokens_per_run", ""),
                    data.get("avg_generate_seconds", ""),
                    data.get("generated_tokens_per_second", ""),
                    data.get("mlc_effective_decode_tokens_per_second", ""),
                ]
            )
        )
    by_name = {name: data for name, data in rows}
    if "MLC" in by_name and "vLLM" in by_name:
        mlc_grid = by_name["MLC"].get("image_grid")
        vllm_grid = by_name["vLLM"].get("image_grid")
        if mlc_grid and vllm_grid:
            if mlc_grid != vllm_grid:
                print(f"image_grid_mismatch mlc={mlc_grid} vllm={vllm_grid}")
                if not allow_image_grid_mismatch:
                    raise RuntimeError(
                        "MLC and vLLM reported different image grids. "
                        "Use --allow-image-grid-mismatch only for intentional diagnostics."
                    )
            else:
                print(f"image_grid_match={mlc_grid}")
        elif comparable_benchmark:
            print(f"image_grid_missing mlc={mlc_grid or ''} vllm={vllm_grid or ''}")
            if not allow_image_grid_mismatch:
                raise RuntimeError(
                    "MLC and vLLM must both report image grids for comparable benchmarks. "
                    "Use --allow-image-grid-mismatch only for intentional diagnostics."
                )
        mlc_tokens = by_name["MLC"].get("generated_tokens_per_run")
        vllm_tokens = by_name["vLLM"].get("generated_tokens_per_run")
        if mlc_tokens and vllm_tokens:
            if mlc_tokens != vllm_tokens:
                print(f"generated_token_count_mismatch mlc={mlc_tokens} vllm={vllm_tokens}")
                if not allow_token_count_mismatch:
                    raise RuntimeError(
                        "MLC and vLLM reported different generated token counts. "
                        "Use --allow-token-count-mismatch only for intentional diagnostics."
                    )
            else:
                print(f"generated_token_count_match={mlc_tokens}")
        elif comparable_benchmark:
            print(
                "generated_token_count_missing "
                f"mlc={mlc_tokens or ''} vllm={vllm_tokens or ''}"
            )
            if not allow_token_count_mismatch:
                raise RuntimeError(
                    "MLC and vLLM must both report generated token counts for comparable "
                    "benchmarks. Use --allow-token-count-mismatch only for intentional "
                    "diagnostics."
                )
        if not comparable_benchmark:
            print(
                "comparison_warning=short_or_cold_run "
                "reason=max_tokens>=128,warmup_runs>=1,benchmark_runs>=5 required for "
                "reported MLC-vLLM throughput deltas"
            )
            return
        mlc_tps = float(by_name["MLC"]["generated_tokens_per_second"])
        vllm_tps = float(by_name["vLLM"]["generated_tokens_per_second"])
        delta = vllm_tps - mlc_tps
        pct = 100.0 * delta / vllm_tps if vllm_tps else 0.0
        print(f"delta_vllm_minus_mlc_tokens_per_second={delta:.3f}")
        print(f"mlc_gap_vs_vllm_percent={pct:.2f}")


def main() -> None:
    args = parse_args()
    args.benchmark_runs = max(2, args.benchmark_runs)
    scheduled = []
    if args.engine in ("both", "mlc"):
        scheduled.append(("MLC", *mlc_command(args)))
    if args.engine in ("both", "vllm"):
        scheduled.append(("vLLM", *vllm_command(args)))
    if args.vllm_first:
        scheduled.reverse()
    rows = [run_command(name, command, env) for name, command, env in scheduled]
    print_summary(
        rows,
        args.allow_image_grid_mismatch,
        args.allow_token_count_mismatch,
        is_comparable_benchmark(args),
    )


if __name__ == "__main__":
    main()
