#!/usr/bin/env python3
"""Run focused steady-state MLC benchmarks for Qwen3.5 image support."""

import argparse
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    "dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC"
)
DEFAULT_MODEL_LIB = (
    "dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-combinedkvcross-chunkedgdn-cg-skipprefill-cuda.so"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MLC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-lib", default=os.environ.get("MLC_MODEL_LIB", DEFAULT_MODEL_LIB))
    parser.add_argument("--image", default="kentucky.png")
    parser.add_argument("--prompt", default="What is in the image?")
    parser.add_argument("--fit-image-width", type=int, default=640)
    parser.add_argument("--fit-image-height", type=int, default=480)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    return parser.parse_args()


def parse_key_values(line: str) -> dict[str, str]:
    values = {}
    for match in re.finditer(r"([a-zA-Z_]+)=([^ ]+)", line):
        values[match.group(1)] = match.group(2)
    return values


def run_case(args: argparse.Namespace, max_tokens: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MLC_MODEL"] = str((REPO_ROOT / args.model).resolve())
    env["MLC_MODEL_LIB"] = str((REPO_ROOT / args.model_lib).resolve())
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
        str(max_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--benchmark-runs",
        str(args.benchmark_runs),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    benchmark = {}
    metrics = {}
    image = {}
    for line in result.stdout.splitlines():
        if line.startswith("image_grid="):
            image = parse_key_values(line)
        elif line.startswith("benchmark "):
            benchmark = parse_key_values(line)
        elif line.startswith("mlc_engine_metrics "):
            metrics = parse_key_values(line)
    if not benchmark or not metrics:
        print(result.stdout)
        raise RuntimeError(f"Could not parse benchmark output for max_tokens={max_tokens}")
    return {"max_tokens": str(max_tokens), **image, **benchmark, **metrics}


def main() -> None:
    args = parse_args()
    rows = [run_case(args, max_tokens) for max_tokens in args.max_tokens]
    columns = [
        "max_tokens",
        "image_embed",
        "generated_tokens_per_run",
        "avg_generate_seconds",
        "generated_tokens_per_second",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_seconds",
        "decode_tokens_per_second",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(row.get(column, "") for column in columns))


if __name__ == "__main__":
    main()
