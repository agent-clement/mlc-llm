#!/usr/bin/env python3
"""Measure a GPU-resident greedy decode loop for the Qwen3.5 image model."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from run_im import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_LIB,
    image_file_to_url,
    load_tvm_module,
    qwen35_assistant_prefix,
    qwen35_position_ids,
    qwen35_user_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MLC_MODEL", str(DEFAULT_MODEL)))
    parser.add_argument("--model-lib", default=os.environ.get("MLC_MODEL_LIB", str(DEFAULT_MODEL_LIB)))
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="What is in the image?")
    parser.add_argument("--fit-image-size", type=int, default=0)
    parser.add_argument("--fit-image-width", type=int, default=0)
    parser.add_argument("--fit-image-height", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    return parser.parse_args()


def build_prompt_embedding(vm, params, device, model_config, model: Path, args: argparse.Namespace):
    import tvm
    from tvm.runtime import ShapeTuple

    from mlc_llm.serve import data
    from mlc_llm.tokenizers import Tokenizer

    tokenizer = Tokenizer(str(model))
    image_url = image_file_to_url(
        args.image,
        fit_size=args.fit_image_size,
        fit_width=args.fit_image_width,
        fit_height=args.fit_image_height,
    )
    image_data = data.ImageData.from_url(image_url, model_config)
    grid = image_data.get_grid_thw()
    if grid is None:
        raise RuntimeError("ImageData did not expose Qwen3.5 grid metadata.")

    prefix_ids = tokenizer.encode(qwen35_user_prefix() + "<|vision_start|>")
    tail_text = f"<|vision_end|>{args.prompt}{qwen35_assistant_prefix()}"
    tail_ids = tokenizer.encode(tail_text)

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

    full_embed = np.concatenate([prefix_embed, image_embed, tail_embed], axis=0).astype("float16")
    position_ids, mrope_delta = qwen35_position_ids(len(prefix_ids), grid, len(tail_ids), 0)
    position_ids = position_ids.astype("int32")
    return full_embed, position_ids, int(mrope_delta), tokenizer, grid


def create_states(vm, model_config):
    import tvm
    from tvm.runtime import ShapeTuple

    kv_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    kv_cache = vm["create_tir_paged_kv_cache"](
        ShapeTuple([1]),
        ShapeTuple([model_config["model_config"].get("context_window_size", 2048)]),
        ShapeTuple([model_config["model_config"].get("prefill_chunk_size", 320)]),
        ShapeTuple([16]),
        ShapeTuple([0]),
    )
    rnn_state = vm["create_rnn_state"](ShapeTuple([1]), ShapeTuple([1]))
    for state in (kv_cache, rnn_state):
        kv_add_sequence(state, 0)
    return kv_cache, rnn_state


def run_prefill(vm, params, device, model_config, full_embed, position_ids, mrope_delta):
    import tvm
    from tvm.runtime import ShapeTuple

    kv_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    kv_cache, rnn_state = create_states(vm, model_config)

    embed_dev = tvm.runtime.tensor(full_embed.reshape(1, *full_embed.shape), device)
    pos_dev = tvm.runtime.tensor(position_ids[:, :, : full_embed.shape[0]], device)
    delta_dev = tvm.runtime.tensor(np.array([[mrope_delta]], dtype="int32"), device)
    logit_positions = tvm.runtime.tensor(np.array([full_embed.shape[0] - 1], dtype="int32"), device)

    for state in (kv_cache, rnn_state):
        kv_begin_forward(state, ShapeTuple([0]), ShapeTuple([full_embed.shape[0]]))
    output = vm["batch_prefill_mrope"](
        embed_dev,
        pos_dev,
        delta_dev,
        logit_positions,
        kv_cache,
        rnn_state,
        params,
    )
    for state in (kv_cache, rnn_state):
        kv_end_forward(state)
    return output[0], kv_cache, rnn_state, delta_dev


def run_device_decode_loop(vm, params, device, logits, kv_cache, rnn_state, delta_dev, max_tokens: int):
    import tvm
    from tvm.runtime import ShapeTuple

    kv_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    token_arrays = []
    for _ in range(max_tokens):
        logits_2d = logits._create_view((1, logits.shape[-1]), "float32")
        token_ids = vm["argmax_logits"](logits_2d)
        token_arrays.append(token_ids)
        embed = vm["embed"](token_ids, params)._create_view((1, 1, 1024), "float16")
        for state in (kv_cache, rnn_state):
            kv_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
        output = vm["decode_mrope"](embed, delta_dev, kv_cache, rnn_state, params)
        for state in (kv_cache, rnn_state):
            kv_end_forward(state)
        logits = output[0]
    device.sync()
    return token_arrays


def main() -> None:
    args = parse_args()
    model = Path(args.model).expanduser()
    model_lib = Path(args.model_lib).expanduser()
    with (model / "mlc-chat-config.json").open(encoding="utf-8") as file:
        model_config = json.load(file)

    vm, params, device = load_tvm_module(model, model_lib)
    try:
        vm["argmax_logits"]
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise RuntimeError("Compiled model does not expose argmax_logits.") from err

    full_embed, position_ids, mrope_delta, tokenizer, grid = build_prompt_embedding(
        vm, params, device, model_config, model, args
    )
    print(
        f"image_grid={grid} prompt_tokens={full_embed.shape[0]} "
        f"mrope_delta={mrope_delta} max_tokens={args.max_tokens}"
    )

    for _ in range(max(0, args.warmup_runs)):
        logits, kv_cache, rnn_state, delta_dev = run_prefill(
            vm, params, device, model_config, full_embed, position_ids, mrope_delta
        )
        run_device_decode_loop(
            vm, params, device, logits, kv_cache, rnn_state, delta_dev, args.max_tokens
        )

    times = []
    final_tokens = None
    for _ in range(max(1, args.benchmark_runs)):
        logits, kv_cache, rnn_state, delta_dev = run_prefill(
            vm, params, device, model_config, full_embed, position_ids, mrope_delta
        )
        device.sync()
        start = time.perf_counter()
        final_tokens = run_device_decode_loop(
            vm, params, device, logits, kv_cache, rnn_state, delta_dev, args.max_tokens
        )
        times.append(time.perf_counter() - start)

    token_ids = [int(token.numpy()[0]) for token in final_tokens or []]
    print(tokenizer.decode(token_ids))
    total_seconds = sum(times)
    total_tokens = args.max_tokens * max(1, args.benchmark_runs)
    print(
        "device_decode_loop "
        f"warmup_runs={max(0, args.warmup_runs)} "
        f"measured_runs={max(1, args.benchmark_runs)} "
        f"tokens_per_run={args.max_tokens} "
        f"total_decode_seconds={total_seconds:.6f} "
        f"avg_decode_seconds={total_seconds / max(1, args.benchmark_runs):.6f} "
        f"decode_tokens_per_second={total_tokens / total_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
