#!/usr/bin/env python3
"""Run the compiled Qwen3.5 image model with MLC."""

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    REPO_ROOT
    / "dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-q0f16-ctx2048-pc320-MLC"
)
DEFAULT_MODEL_LIB = (
    REPO_ROOT
    / "dist/libs/logos-multitask-qwen3.5-2026-05-03-best-q0f16-ctx2048-pc320-explicit-paged-stabledecodeinput-cuda.so"
)


def configure_repo_env() -> None:
    python_path = REPO_ROOT / "python"
    if str(python_path) not in sys.path:
        sys.path.insert(0, str(python_path))
    os.environ.setdefault("MLC_LIBRARY_PATH", str(REPO_ROOT / "build-mlc-env"))


def image_to_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:image")):
        return image

    path = Path(image).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_file_to_url(
    image: str,
    fit_size: int = 0,
    square_canvas: bool = True,
    fit_width: int = 0,
    fit_height: int = 0,
) -> str:
    if image.startswith(("http://", "https://", "data:image")):
        if fit_size or fit_width or fit_height:
            raise ValueError("Image resizing only supports local image files.")
        return image_to_url(image)

    path = Path(image).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if fit_size <= 0 and (fit_width <= 0 or fit_height <= 0):
        return image_to_url(image)

    from io import BytesIO

    from PIL import Image

    source = Image.open(path).convert("RGB")
    if fit_width > 0 and fit_height > 0:
        target_width = fit_width
        target_height = fit_height
        source.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        x = (target_width - source.width) // 2
        y = (target_height - source.height) // 2
        canvas.paste(source, (x, y))
    else:
        source.thumbnail((fit_size, fit_size), Image.Resampling.LANCZOS)

    if fit_width <= 0 and fit_height <= 0 and square_canvas:
        canvas = Image.new("RGB", (fit_size, fit_size), (255, 255, 255))
        x = (fit_size - source.width) // 2
        y = (fit_size - source.height) // 2
        canvas.paste(source, (x, y))
    elif fit_width <= 0 and fit_height <= 0:
        canvas = source

    output = BytesIO()
    canvas.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MLC_MODEL", str(DEFAULT_MODEL)))
    parser.add_argument(
        "--model-lib",
        default=os.environ.get("MLC_MODEL_LIB", str(DEFAULT_MODEL_LIB)),
    )
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--image", help="Local image path, HTTP URL, or data:image URL.")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument(
        "--fit-image-size",
        type=int,
        default=0,
        help="Resize a local image into a square canvas before inference.",
    )
    parser.add_argument(
        "--fit-image-max-side",
        type=int,
        default=0,
        help="Resize a local image preserving aspect ratio before inference.",
    )
    parser.add_argument(
        "--fit-image-width",
        type=int,
        default=0,
        help="Resize a local image into a fixed-width canvas before inference.",
    )
    parser.add_argument(
        "--fit-image-height",
        type=int,
        default=0,
        help="Resize a local image into a fixed-height canvas before inference.",
    )
    parser.add_argument(
        "--no-auto-resize",
        action="store_true",
        help="Do not automatically resize local images to fit the compiled prefill chunk.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS/stop token IDs. Useful for fixed-length decode benchmarking.",
    )
    parser.add_argument("--max-total-sequence-length", type=int)
    parser.add_argument(
        "--kv-cache-page-size",
        type=int,
        default=int(os.environ.get("MLC_KV_CACHE_PAGE_SIZE", "16")),
        help="KV cache page size. Non-16 values require a library compiled with the same page size.",
    )
    parser.add_argument(
        "--recompute-prefill",
        action="store_true",
        help="Debug fallback: recompute full prefill for each generated token.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="Run generate this many measured times after model load.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run generate this many unmeasured warmup times.",
    )
    parser.add_argument(
        "--trace-summary",
        action="store_true",
        help="Print an event-trace timing summary for MLC engine phases.",
    )
    parser.add_argument(
        "--trace-json",
        help="Write the raw MLC event trace in Chrome trace JSON format.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Use cudaProfilerStart/Stop around measured generate calls for Nsight capture.",
    )
    parser.add_argument(
        "--generate-timing",
        action="store_true",
        help="Print SyncMLCEngine.generate timing breakdown for benchmark runs.",
    )
    parser.add_argument(
        "--sync-decode-timing",
        action="store_true",
        help=(
            "Synchronize after decode subphases before recording MLC engine timing metrics. "
            "This is slower but makes model/probs/sample attribution more precise."
        ),
    )
    parser.add_argument(
        "--prefix-cache-mode",
        choices=("radix", "disable"),
        default="radix",
        help="Prefix cache mode used by the MLC engine.",
    )
    parser.add_argument(
        "--prefill-mode",
        choices=("chunked", "hybrid"),
        default="hybrid",
        help="MLC prefill scheduling mode.",
    )
    return parser.parse_args()


def qwen35_user_prefix() -> str:
    return "<|im_start|>user\n"


def qwen35_assistant_prefix() -> str:
    return "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def summarize_trace_events(trace_json: str) -> str:
    events = json.loads(trace_json)
    starts = {}
    totals = {}
    counts = {}

    def normalize_name(name: str) -> str:
        marker = " ("
        if name.endswith(")") and marker in name:
            base, suffix = name.rsplit(marker, 1)
            if suffix[:-1].isdigit():
                return base
        return name

    for event in events:
        name = event.get("name")
        phase = event.get("ph")
        tid = event.get("tid")
        ts = event.get("ts")
        if name is None or phase is None or tid is None or ts is None:
            continue
        name = normalize_name(name)
        key = (tid, name)
        if phase == "B":
            starts.setdefault(key, []).append(ts)
        elif phase == "E":
            stack = starts.get(key)
            if not stack:
                continue
            begin = stack.pop()
            duration_us = max(0, ts - begin)
            totals[name] = totals.get(name, 0) + duration_us
            counts[name] = counts.get(name, 0) + 1

    if not totals:
        return "mlc_trace_summary no_paired_events=1"

    parts = ["mlc_trace_summary"]
    for name, total_us in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        safe_name = name.replace(" ", "_")
        parts.append(
            f"{safe_name}_ms={total_us / 1000.0:.3f} "
            f"{safe_name}_count={counts.get(name, 0)}"
        )
    return " ".join(parts)


def load_tvm_module(model: Path, model_lib: Path):
    import tvm
    from tvm import relax
    from tvm.contrib import tvmjs
    from tvm.runtime.vm import VirtualMachine

    device = tvm.runtime.device("cuda", 0)
    executable = tvm.runtime.load_module(str(model_lib))
    vm = relax.VirtualMachine(executable, device).module
    metadata = json.loads(VirtualMachine(executable, tvm.runtime.device("cpu"))["_metadata"]())
    params_by_name, meta = tvmjs.load_tensor_cache(str(model), device)
    param_names = [param["name"] for param in metadata["params"]]
    if len(param_names) != meta["ParamSize"]:
        raise RuntimeError(
            f"Parameter metadata mismatch: {len(param_names)} names vs {meta['ParamSize']} tensors"
        )
    params = [params_by_name[name] for name in param_names]
    return vm, params, device


def qwen35_position_ids(prefix_len: int, image_grid, tail_len: int, generated_len: int):
    import numpy as np

    grid_t, grid_h, grid_w = image_grid
    merge = 2
    llm_grid_t = grid_t
    llm_grid_h = grid_h // merge
    llm_grid_w = grid_w // merge
    total_len = prefix_len + llm_grid_t * llm_grid_h * llm_grid_w + tail_len + generated_len
    position_ids = np.zeros((3, 1, total_len), dtype="int32")

    offset = 0
    current_pos = 0
    for i in range(prefix_len):
        position_ids[:, 0, offset + i] = current_pos + i
    offset += prefix_len
    current_pos += prefix_len

    for t in range(llm_grid_t):
        for h in range(llm_grid_h):
            for w in range(llm_grid_w):
                position_ids[0, 0, offset] = current_pos + t
                position_ids[1, 0, offset] = current_pos + h
                position_ids[2, 0, offset] = current_pos + w
                offset += 1
    current_pos += max(grid_h, grid_w) // merge

    for i in range(tail_len + generated_len):
        position_ids[:, 0, offset + i] = current_pos + i

    return position_ids, int(position_ids.max() + 1 - total_len)


def run_qwen35_image_correctness_path(
    model: Path,
    model_lib: Path,
    model_config: dict,
    prompt_text: str,
    image_data,
    max_tokens: int,
):
    """Greedy Qwen3.5 image generation via repeated full prefill for diagnostics."""

    import numpy as np
    import tvm
    from tvm.runtime import ShapeTuple

    from mlc_llm.tokenizers import Tokenizer

    tokenizer = Tokenizer(str(model))

    prefix_ids = tokenizer.encode(qwen35_user_prefix() + "<|vision_start|>")
    tail_text = f"<|vision_end|>{prompt_text}{qwen35_assistant_prefix()}"
    tail_ids = tokenizer.encode(tail_text)
    grid = image_data.get_grid_thw()
    if grid is None:
        raise RuntimeError("Qwen3.5 correctness path requires image grid metadata.")
    prompt_tokens_with_generation = len(prefix_ids) + len(image_data) + len(tail_ids) + max_tokens
    prefill_chunk_size = int(model_config["model_config"].get("prefill_chunk_size", 0))
    if prefill_chunk_size and prompt_tokens_with_generation > prefill_chunk_size:
        for candidate in (512, 1024, 2048):
            if candidate < prompt_tokens_with_generation or candidate <= prefill_chunk_size:
                continue
            candidate_model = Path(
                str(model).replace(f"-pc{prefill_chunk_size}-", f"-pc{candidate}-")
            )
            candidate_lib = Path(
                str(model_lib).replace(f"-pc{prefill_chunk_size}-", f"-pc{candidate}-")
            )
            candidate_config = candidate_model / "mlc-chat-config.json"
            if candidate_config.is_file() and candidate_lib.is_file():
                model = candidate_model
                model_lib = candidate_lib
                with candidate_config.open(encoding="utf-8") as file:
                    model_config = json.load(file)
                tokenizer = Tokenizer(str(model))
                prefix_ids = tokenizer.encode(qwen35_user_prefix() + "<|vision_start|>")
                tail_ids = tokenizer.encode(tail_text)
                break

    vm, params, device = load_tvm_module(model, model_lib)

    prefix_embed = vm["embed"](
        tvm.runtime.tensor(np.array(prefix_ids, dtype="int32"), device), params
    ).numpy()
    tail_embed = vm["embed"](
        tvm.runtime.tensor(np.array(tail_ids, dtype="int32"), device), params
    ).numpy()

    patch_size = int(model_config["model_config"]["vision_config"].get("patch_size", 16))
    resized_height = grid[1] * patch_size
    resized_width = grid[2] * patch_size
    image_tensor = image_data.image if image_data.image.device == device else image_data.image.copyto(device)
    image_embed = vm["image_embed"](
        image_tensor,
        ShapeTuple([resized_height]),
        ShapeTuple([resized_width]),
        ShapeTuple([grid[1]]),
        ShapeTuple([grid[2]]),
        params,
    ).numpy()

    kv_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    kv_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    generated_ids = []
    generated_embeds = []
    stop_token_ids = {248046, 248044}

    for _ in range(max_tokens):
        if generated_embeds:
            tail_generated_embed = np.concatenate([tail_embed, *generated_embeds], axis=0)
        else:
            tail_generated_embed = tail_embed
        position_ids, mrope_delta = qwen35_position_ids(
            len(prefix_ids), grid, len(tail_ids), len(generated_ids)
        )

        kv_cache = vm["create_tir_paged_kv_cache"](
            ShapeTuple([1]),
            ShapeTuple([model_config["model_config"].get("context_window_size", 2048)]),
            ShapeTuple([model_config["model_config"].get("prefill_chunk_size", 512)]),
            ShapeTuple([16]),
            ShapeTuple([0]),
        )
        rnn_state = vm["create_rnn_state"](ShapeTuple([1]), ShapeTuple([1]))
        for state in (kv_cache, rnn_state):
            kv_add_sequence(state, 0)

        full_embed = np.concatenate([prefix_embed, image_embed, tail_generated_embed], axis=0)
        prefill_chunk_size = int(model_config["model_config"].get("prefill_chunk_size", 0))
        if prefill_chunk_size <= 0 or full_embed.shape[0] <= prefill_chunk_size:
            chunk_specs = [(full_embed, position_ids)]
        else:
            chunk_specs = [
                (prefix_embed, position_ids[:, :, : len(prefix_ids)]),
                (
                    image_embed,
                    position_ids[
                        :,
                        :,
                        len(prefix_ids) : len(prefix_ids) + image_embed.shape[0],
                    ],
                ),
                (
                    tail_generated_embed,
                    position_ids[:, :, len(prefix_ids) + image_embed.shape[0] :],
                ),
            ]
        output = None
        for chunk_embed, chunk_position_ids in chunk_specs:
            chunk_embed = chunk_embed.astype("float16")
            for state in (kv_cache, rnn_state):
                kv_begin_forward(state, ShapeTuple([0]), ShapeTuple([chunk_embed.shape[0]]))
            output = vm["batch_prefill_mrope"](
                tvm.runtime.tensor(chunk_embed.reshape(1, *chunk_embed.shape), device),
                tvm.runtime.tensor(chunk_position_ids.astype("int32"), device),
                tvm.runtime.tensor(np.array([[mrope_delta]], dtype="int32"), device),
                tvm.runtime.tensor(np.array([chunk_embed.shape[0] - 1], dtype="int32"), device),
                kv_cache,
                rnn_state,
                params,
            )
            for state in (kv_cache, rnn_state):
                kv_end_forward(state)

        logits = output[0].numpy()[0, 0]
        next_token = int(np.argmax(logits))
        if next_token in stop_token_ids:
            break
        generated_ids.append(next_token)
        generated_embeds.append(
            vm["embed"](tvm.runtime.tensor(np.array([next_token], dtype="int32"), device), params)
            .numpy()
            .reshape(1, -1)
        )

    return tokenizer.decode(generated_ids)


def main() -> None:
    configure_repo_env()

    from mlc_llm.protocol.debug_protocol import DebugConfig
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve import data
    from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine

    args = parse_args()
    if args.generate_timing:
        os.environ["MLC_SYNC_GENERATE_TIMING"] = "1"
    if args.sync_decode_timing:
        os.environ["MLC_SYNC_DECODE_TIMING"] = "1"
    preprocess_start = time.perf_counter()
    fixed_image_size = args.fit_image_width > 0 or args.fit_image_height > 0
    if fixed_image_size and (args.fit_image_width <= 0 or args.fit_image_height <= 0):
        raise ValueError("Use --fit-image-width and --fit-image-height together.")
    if sum(bool(value) for value in (args.fit_image_size, args.fit_image_max_side, fixed_image_size)) > 1:
        raise ValueError(
            "Use only one of --fit-image-size, --fit-image-max-side, "
            "or --fit-image-width/--fit-image-height."
        )

    model = Path(args.model).expanduser()
    model_lib = Path(args.model_lib).expanduser()
    config_path = model / "mlc-chat-config.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot find mlc-chat-config.json under {model}")
    if not model_lib.is_file():
        raise FileNotFoundError(f"Cannot find model library: {model_lib}")

    with config_path.open(encoding="utf-8") as file:
        model_config = json.load(file)

    max_total_sequence_length = (
        args.max_total_sequence_length
        or model_config["model_config"].get("context_window_size")
        or 2048
    )

    prompt = [data.TextData(args.prompt if args.raw_prompt else qwen35_user_prefix())]
    if args.image:
        explicit_fit_size = args.fit_image_size or args.fit_image_max_side
        square_canvas = args.fit_image_size > 0
        image_url = image_file_to_url(
            args.image,
            explicit_fit_size,
            square_canvas,
            args.fit_image_width,
            args.fit_image_height,
        )
        image_data = data.ImageData.from_url(
            image_url, model_config
        )
        image_embed = len(image_data)
        prefill_chunk_size = int(model_config["model_config"].get("prefill_chunk_size", 0))
        if (
            prefill_chunk_size
            and image_embed > prefill_chunk_size
            and explicit_fit_size <= 0
            and not fixed_image_size
            and not args.no_auto_resize
            and not args.image.startswith(("http://", "https://", "data:image"))
        ):
            patch_size = int(
                model_config["model_config"]["vision_config"].get("patch_size", 16)
            )
            merge_size = int(
                model_config["model_config"]["vision_config"].get("spatial_merge_size", 2)
            )
            fit_size = int(math.floor(math.sqrt(prefill_chunk_size)) * patch_size * merge_size)
            image_url = image_file_to_url(args.image, fit_size, square_canvas=False)
            image_data = data.ImageData.from_url(image_url, model_config)
            image_embed = len(image_data)
            print(
                f"auto_resized_image_max_side={fit_size} "
                f"image_grid={image_data.get_grid_thw()} image_embed={image_embed}"
            )
        else:
            print(f"image_grid={image_data.get_grid_thw()} image_embed={image_embed}")

        if prefill_chunk_size and image_embed > prefill_chunk_size:
            required = int(math.ceil(image_embed / 64) * 64)
            raise SystemExit(
                "Image embedding is larger than this compiled model's prefill chunk. "
                f"image_embed={image_embed}, prefill_chunk_size={prefill_chunk_size}. "
                "Recompile with a larger chunk, for example:\n"
                f"  PREFILL_CHUNK_SIZE={required} ./compile_im.sh\n"
                "Then run with:\n"
                "  MLC_MODEL=dist/logos-multitask-qwen3.5-2026-05-03-best-fusedinproj-"
                f"q0f16-ctx2048-pc{required}-MLC "
                "MLC_MODEL_LIB=dist/libs/logos-multitask-qwen3.5-2026-05-03-best-"
                f"q0f16-ctx2048-pc{required}-explicit-paged-stabledecodeinput-cuda.so "
                "./run_im.sh ..."
            )
        if args.raw_prompt:
            prompt.append(image_data)
        else:
            prompt.append(data.TextData("<|vision_start|>"))
            prompt.append(image_data)
            prompt.append(data.TextData(f"<|vision_end|>{args.prompt}{qwen35_assistant_prefix()}"))

        if (
            args.recompute_prefill
            and model_config.get("model_type") == "qwen3_5"
            and args.temperature == 0.0
            and args.top_p == 1.0
            and args.frequency_penalty == 0.0
            and args.presence_penalty == 0.0
            and args.repetition_penalty == 1.0
            and not args.raw_prompt
        ):
            print(
                run_qwen35_image_correctness_path(
                    model, model_lib, model_config, args.prompt, image_data, args.max_tokens
                )
            )
            return
    elif not args.raw_prompt:
        prompt.append(data.TextData(f"{args.prompt}{qwen35_assistant_prefix()}"))

    preprocess_seconds = time.perf_counter() - preprocess_start
    engine_init_start = time.perf_counter()
    engine = SyncMLCEngine(
        model=str(model),
        model_lib=str(model_lib),
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=max_total_sequence_length,
            max_num_sequence=1,
            kv_cache_page_size=args.kv_cache_page_size,
            prefix_cache_mode=args.prefix_cache_mode,
            prefill_mode=args.prefill_mode,
        ),
        enable_tracing=bool(args.trace_summary or args.trace_json),
    )
    engine_init_seconds = time.perf_counter() - engine_init_start

    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=[248046, 248044],
        debug_config=DebugConfig(ignore_eos=args.ignore_eos),
    )
    benchmark_runs = max(1, args.benchmark_runs)
    warmup_runs = max(0, args.warmup_runs)

    for _ in range(warmup_runs):
        engine.generate([prompt], generation_config)
    if warmup_runs:
        engine.reset()

    profiler_start = profiler_stop = None
    if args.cuda_profiler_range:
        import tvm

        profiler_start = tvm.get_global_func("mlc.debug_cuda_profiler_start")
        profiler_stop = tvm.get_global_func("mlc.debug_cuda_profiler_stop")
        profiler_start()

    times = []
    generate_timings = []
    outputs = None
    try:
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            outputs, _ = engine.generate([prompt], generation_config)
            times.append(time.perf_counter() - start)
            if args.generate_timing:
                generate_timings.append(dict(getattr(engine, "last_generate_timing", {})))
    finally:
        if profiler_stop is not None:
            profiler_stop()

    output_text = outputs[0][0]
    metrics = engine.metrics().metrics
    print(output_text)
    if benchmark_runs > 1 or warmup_runs > 0:
        generated_tokens = len(engine.tokenizer.encode(output_text))
        total_seconds = sum(times)
        prefill_seconds = metrics.get("engine_prefill_time_sum", 0.0)
        decode_seconds = metrics.get("engine_decode_time_sum", 0.0)
        step_seconds = metrics.get("engine_step_time_sum", 0.0)
        action_step_seconds = metrics.get("engine_action_step_time_sum", 0.0)
        postprocess_seconds = metrics.get("engine_postprocess_time_sum", 0.0)
        action_time_by_index = metrics.get("engine_action_time_by_index", {})
        batch_decode_action_seconds = metrics.get("engine_batch_decode_action_time_sum", 0.0)
        batch_decode_prepare_seconds = metrics.get("engine_batch_decode_prepare_time_sum", 0.0)
        batch_decode_deferred_commit_seconds = metrics.get(
            "engine_batch_decode_deferred_commit_time_sum", 0.0
        )
        batch_decode_deferred_commit_copy_sync_seconds = metrics.get(
            "engine_batch_decode_deferred_commit_copy_sync_time_sum", 0.0
        )
        batch_decode_deferred_commit_cpu_seconds = metrics.get(
            "engine_batch_decode_deferred_commit_cpu_time_sum", 0.0
        )
        batch_decode_model_seconds = metrics.get("engine_batch_decode_model_time_sum", 0.0)
        batch_decode_logits_update_seconds = metrics.get(
            "engine_batch_decode_logits_update_time_sum", 0.0
        )
        batch_decode_probs_seconds = metrics.get("engine_batch_decode_probs_time_sum", 0.0)
        batch_decode_sample_seconds = metrics.get("engine_batch_decode_sample_time_sum", 0.0)
        batch_decode_device_token_copy_seconds = metrics.get(
            "engine_batch_decode_device_token_copy_time_sum", 0.0
        )
        prefill_tokens = metrics.get("prefill_tokens_sum", 0)
        decode_tokens = metrics.get("decode_tokens_sum", 0)
        print(
            "benchmark "
            f"warmup_runs={warmup_runs} measured_runs={benchmark_runs} "
            f"generated_tokens_per_run={generated_tokens} "
            f"preprocess_seconds={preprocess_seconds:.6f} "
            f"engine_init_seconds={engine_init_seconds:.6f} "
            f"total_generate_seconds={total_seconds:.6f} "
            f"avg_generate_seconds={total_seconds / benchmark_runs:.6f} "
            f"generated_tokens_per_second={(generated_tokens * benchmark_runs) / total_seconds:.3f}"
        )
        print(
            "mlc_runtime_config "
            f"defer_cpu_token_burst={os.environ.get('MLC_DEFER_CPU_TOKEN_BURST', '')} "
            f"decode_burst_steps={os.environ.get('MLC_DECODE_BURST_STEPS', '')} "
            f"stable_decode_embedding={os.environ.get('MLC_STABLE_DECODE_EMBEDDING', '')} "
            f"kv_cache_page_size={args.kv_cache_page_size} "
            f"sync_decode_timing={os.environ.get('MLC_SYNC_DECODE_TIMING', '')}"
        )
        print(
            "mlc_engine_metrics "
            f"prefill_tokens={prefill_tokens} "
            f"prefill_seconds={prefill_seconds:.6f} "
            f"prefill_tokens_per_second="
            f"{prefill_tokens / prefill_seconds if prefill_seconds else 0.0:.3f} "
            f"decode_tokens={decode_tokens} "
            f"decode_seconds={decode_seconds:.6f} "
            f"decode_tokens_per_second="
            f"{decode_tokens / decode_seconds if decode_seconds else 0.0:.3f}"
        )
        effective_decode_seconds = decode_seconds + batch_decode_deferred_commit_copy_sync_seconds
        if batch_decode_deferred_commit_copy_sync_seconds:
            print(
                "mlc_effective_decode_metrics "
                f"decode_tokens={decode_tokens} "
                f"decode_seconds={decode_seconds:.6f} "
                f"deferred_commit_copy_sync_seconds="
                f"{batch_decode_deferred_commit_copy_sync_seconds:.6f} "
                f"effective_decode_seconds={effective_decode_seconds:.6f} "
                f"effective_decode_tokens_per_second="
                f"{decode_tokens / effective_decode_seconds if effective_decode_seconds else 0.0:.3f}"
            )
        print(
            "mlc_engine_step_metrics "
            f"step_seconds={step_seconds:.6f} "
            f"action_step_seconds={action_step_seconds:.6f} "
            f"postprocess_seconds={postprocess_seconds:.6f} "
            f"unclassified_step_seconds="
            f"{max(0.0, step_seconds - action_step_seconds - postprocess_seconds):.6f}"
        )
        if action_time_by_index:
            action_parts = []
            for key, value in sorted(action_time_by_index.items()):
                action_index = key.split("action_index=", 1)[-1].rstrip("}")
                action_parts.append(f"action{action_index}_seconds={float(value):.6f}")
            print("mlc_engine_action_index_metrics " + " ".join(action_parts))
        if batch_decode_action_seconds:
            classified_inner_decode_seconds = (
                batch_decode_model_seconds
                + batch_decode_logits_update_seconds
                + batch_decode_probs_seconds
                + batch_decode_sample_seconds
                + batch_decode_device_token_copy_seconds
            )
            print(
                "mlc_batch_decode_action_metrics "
                f"action_seconds={batch_decode_action_seconds:.6f} "
                f"prepare_seconds={batch_decode_prepare_seconds:.6f} "
                f"inner_decode_seconds={decode_seconds:.6f} "
                f"model_seconds={batch_decode_model_seconds:.6f} "
                f"logits_update_seconds={batch_decode_logits_update_seconds:.6f} "
                f"probs_seconds={batch_decode_probs_seconds:.6f} "
                f"sample_seconds={batch_decode_sample_seconds:.6f} "
                f"device_token_copy_seconds={batch_decode_device_token_copy_seconds:.6f} "
                f"unclassified_inner_decode_seconds="
                f"{max(0.0, decode_seconds - classified_inner_decode_seconds):.6f} "
                f"deferred_commit_seconds={batch_decode_deferred_commit_seconds:.6f} "
                f"deferred_commit_copy_sync_seconds="
                f"{batch_decode_deferred_commit_copy_sync_seconds:.6f} "
                f"deferred_commit_cpu_seconds={batch_decode_deferred_commit_cpu_seconds:.6f} "
                f"other_batch_decode_seconds="
                f"{max(0.0, batch_decode_action_seconds - batch_decode_prepare_seconds - decode_seconds - batch_decode_deferred_commit_seconds):.6f}"
            )
        if args.generate_timing and generate_timings:
            totals = {}
            for timing in generate_timings:
                for key, value in timing.items():
                    if isinstance(value, (int, float)):
                        totals[key] = totals.get(key, 0.0) + value
            total_generate_timing = totals.get("total_generate_seconds", 0.0)
            step_loop_timing = totals.get("step_loop_seconds", 0.0)
            print(
                "sync_generate_timing "
                f"setup_seconds={totals.get('setup_seconds', 0.0):.6f} "
                f"get_callback_seconds={totals.get('get_callback_seconds', 0.0):.6f} "
                f"set_callback_seconds={totals.get('set_callback_seconds', 0.0):.6f} "
                f"add_request_seconds={totals.get('add_request_seconds', 0.0):.6f} "
                f"step_loop_seconds={totals.get('step_loop_seconds', 0.0):.6f} "
                f"callback_seconds={totals.get('callback_seconds', 0.0):.6f} "
                f"detokenize_seconds={totals.get('detokenize_seconds', 0.0):.6f} "
                f"restore_callback_seconds={totals.get('restore_callback_seconds', 0.0):.6f} "
                f"total_generate_seconds={total_generate_timing:.6f} "
                f"outside_step_loop_seconds={max(0.0, total_generate_timing - step_loop_timing):.6f} "
                f"step_count={int(totals.get('step_count', 0))}"
            )
    if args.trace_summary or args.trace_json:
        trace_json = engine.trace_recorder.dump_json()
        if args.trace_json:
            Path(args.trace_json).write_text(trace_json, encoding="utf-8")
        if args.trace_summary:
            print(summarize_trace_events(trace_json))
    engine.reset()


if __name__ == "__main__":
    main()
