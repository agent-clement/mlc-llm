#!/usr/bin/env python3
"""Run the Qwen3.5 image model through vLLM for MLC comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


DEFAULT_HF_MODEL = "familiar-ai/logos-multitask-qwen3.5-2026-05-03-best"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_VLLM_MODEL = (
    REPO_ROOT / "dist/logos-multitask-qwen3.5-2026-05-03-best-vllm-hf-tokenizercompat"
)
PROCESSOR_USE_FAST_CHOICES = ("auto", "true", "false")


def default_model_path() -> str:
    if "HF_MODEL" in os.environ:
        return os.environ["HF_MODEL"]
    if DEFAULT_LOCAL_VLLM_MODEL.exists():
        return str(DEFAULT_LOCAL_VLLM_MODEL)
    return DEFAULT_HF_MODEL


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
    parser.add_argument("--hf-model", default=default_model_path())
    parser.add_argument("--tokenizer", default=os.environ.get("VLLM_TOKENIZER"))
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--image", required=True)
    parser.add_argument("--fit-image-width", type=int, default=640)
    parser.add_argument("--fit-image-height", type=int, default=480)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS/stop token IDs. Useful for fixed-length decode benchmarking.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=None,
        help=(
            "Explicit vLLM KV cache memory budget in bytes. When set, vLLM "
            "uses this instead of sizing KV cache from gpu_memory_utilization."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override vLLM scheduler token budget. Defaults to max_model_len for batch-1.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="Override vLLM KV cache block size. 0 keeps vLLM's backend-selected default.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--mm-encoder-attn-backend", default=None)
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument(
        "--processor-use-fast",
        choices=PROCESSOR_USE_FAST_CHOICES,
        default=processor_use_fast_default(),
        help="Pass use_fast to AutoProcessor. 'auto' preserves Transformers' default.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Use cudaProfilerStart/Stop around measured generate calls for Nsight capture.",
    )
    return parser.parse_args()


def load_image(path: str, fit_width: int, fit_height: int) -> Image.Image:
    from PIL import Image

    image = Image.open(Path(path).expanduser()).convert("RGB")
    if fit_width <= 0 and fit_height <= 0:
        return image
    if fit_width <= 0 or fit_height <= 0:
        raise ValueError("Use --fit-image-width and --fit-image-height together.")
    image.thumbnail((fit_width, fit_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (fit_width, fit_height), (255, 255, 255))
    canvas.paste(image, ((fit_width - image.width) // 2, (fit_height - image.height) // 2))
    return canvas


def prepare_tokenizer_path(model: str, explicit_tokenizer: str | None) -> str | None:
    """Create a local tokenizer metadata copy when vLLM cannot load TokenizersBackend."""
    if explicit_tokenizer:
        return explicit_tokenizer

    model_path = Path(model).expanduser()
    if model_path.exists():
        source = model_path
        cache_name = model_path.name
    else:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(repo_id=model))
        cache_name = model.replace("/", "--")

    tokenizer_config = source / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return None

    config = json.loads(tokenizer_config.read_text())
    if config.get("tokenizer_class") != "TokenizersBackend":
        return str(source)

    output = Path("dist") / "vllm-tokenizers" / f"{cache_name}-qwen2"
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ):
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)
    config["tokenizer_class"] = "Qwen2TokenizerFast"
    (output / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return str(output)


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this vLLM environment. "
            f"torch={torch.__version__}, torch_cuda={torch.version.cuda}. "
            "Install a vLLM/Torch build compatible with the NVIDIA driver or update the driver."
        )

    from vllm import LLM, SamplingParams

    preprocess_start = time.perf_counter()
    image = load_image(args.image, args.fit_image_width, args.fit_image_height)
    processor_kwargs = {"trust_remote_code": True}
    if args.processor_use_fast != "auto":
        processor_kwargs["use_fast"] = args.processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(args.hf_model, **processor_kwargs)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    print(f"image_grid_thw={inputs.get('image_grid_thw')}")
    print(f"input_ids_shape={tuple(inputs['input_ids'].shape)}")
    pixel_values = inputs.get("pixel_values")
    print(f"pixel_values_shape={tuple(pixel_values.shape) if pixel_values is not None else None}")
    preprocess_seconds = time.perf_counter() - preprocess_start

    load_start = time.perf_counter()
    llm_kwargs = {
        "model": args.hf_model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens or args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    tokenizer_path = prepare_tokenizer_path(args.hf_model, args.tokenizer)
    if tokenizer_path:
        llm_kwargs["tokenizer"] = tokenizer_path
    if args.attention_backend:
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.mm_encoder_attn_backend:
        llm_kwargs["mm_encoder_attn_backend"] = args.mm_encoder_attn_backend
    if args.skip_mm_profiling:
        llm_kwargs["skip_mm_profiling"] = True
    if args.block_size > 0:
        llm_kwargs["block_size"] = args.block_size
    llm = LLM(**llm_kwargs)
    cache_config = llm.llm_engine.vllm_config.cache_config
    print(
        "vllm_cache_config "
        f"block_size={cache_config.block_size} "
        f"user_specified_block_size={cache_config.user_specified_block_size} "
        f"mamba_block_size={cache_config.mamba_block_size} "
        f"mamba_cache_mode={cache_config.mamba_cache_mode} "
        f"mamba_page_size_padded={cache_config.mamba_page_size_padded}"
    )
    model_load_seconds = time.perf_counter() - load_start
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    prompt = {"prompt": text, "multi_modal_data": {"image": image}}

    for _ in range(max(0, args.warmup_runs)):
        llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)

    outputs = None
    times = []
    benchmark_runs = max(1, args.benchmark_runs)
    if args.cuda_profiler_range:
        torch.cuda.synchronize()
        err = torch.cuda.cudart().cudaProfilerStart()
        if err != 0:
            raise RuntimeError(f"cudaProfilerStart failed with error code {err}")
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        times.append(time.perf_counter() - start)
    if args.cuda_profiler_range:
        torch.cuda.synchronize()
        err = torch.cuda.cudart().cudaProfilerStop()
        if err != 0:
            raise RuntimeError(f"cudaProfilerStop failed with error code {err}")

    output = outputs[0].outputs[0]
    text_out = output.text
    generated_tokens = len(output.token_ids)
    total_seconds = sum(times)
    print(text_out)
    print(
        "benchmark "
        f"warmup_runs={max(0, args.warmup_runs)} measured_runs={benchmark_runs} "
        f"generated_tokens_per_run={generated_tokens} "
        f"preprocess_seconds={preprocess_seconds:.6f} "
        f"model_load_seconds={model_load_seconds:.6f} "
        f"total_generate_seconds={total_seconds:.6f} "
        f"avg_generate_seconds={total_seconds / benchmark_runs:.6f} "
        f"generated_tokens_per_second={(generated_tokens * benchmark_runs) / total_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
