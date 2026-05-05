#!/usr/bin/env python3
"""Direct VM microprofiles for Qwen3.5 image prefill and decode phases."""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from run_im import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_LIB,
    configure_repo_env,
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
    parser.add_argument("--image", default="kentucky.png")
    parser.add_argument("--prompt", default="What is in the image?")
    parser.add_argument("--fit-image-size", type=int, default=512)
    parser.add_argument(
        "--phase",
        choices=[
            "prefill",
            "decode",
            "decode-hidden",
            "lm-head",
            "decode-state",
            "decode-body",
            "decode-hidden-body",
            "rnn-state-copy",
            "rnn-raw-access",
        ],
        required=True,
        help=(
            "`decode` measures full per-token decode. `decode-hidden` measures decode "
            "through the final hidden state, without lm_head. `lm-head` measures only "
            "get_logits on a cached hidden state. `decode-state` measures only "
            "kv/rnn begin/end bookkeeping. `decode-body` and `decode-hidden-body` measure "
            "repeated VM calls inside one begin_forward window; these are microprofiles "
            "only, not semantically valid generation loops. `rnn-state-copy` measures the "
            "existing RNNState get/set copy boundary for all linear-attention layers. "
            "`rnn-raw-access` measures the proposed raw storage/slot accessor boundary."
        ),
    )
    parser.add_argument("--runs", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=4)
    parser.add_argument("--decode-token-id", type=int, default=262)
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap only the measured loop with cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--reuse-input-tensors",
        action="store_true",
        help=(
            "Reuse device input NDArrays across calls. This is useful for diagnosing "
            "host allocation overhead, but it changes the VM/CUDA graph execution "
            "regime and is not the default decode benchmark."
        ),
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace):
    import tvm
    from tvm.runtime import ShapeTuple

    from mlc_llm.serve import data
    from mlc_llm.tokenizers import Tokenizer

    model = Path(args.model).expanduser()
    model_lib = Path(args.model_lib).expanduser()
    with (model / "mlc-chat-config.json").open(encoding="utf-8") as file:
        model_config = json.load(file)

    vm, params, device = load_tvm_module(model, model_lib)
    tokenizer = Tokenizer(str(model))

    image_url = image_file_to_url(args.image, args.fit_image_size, square_canvas=True)
    image_data = data.ImageData.from_url(image_url, model_config)
    grid = image_data.get_grid_thw()
    if grid is None:
        raise RuntimeError("Expected Qwen3.5 image grid metadata.")

    prefix_ids = tokenizer.encode(qwen35_user_prefix() + "<|vision_start|>")
    tail_ids = tokenizer.encode(
        f"<|vision_end|>{args.prompt}{qwen35_assistant_prefix()}"
    )

    prefix_embed = vm["embed"](
        tvm.runtime.tensor(np.array(prefix_ids, dtype="int32"), device), params
    ).numpy()
    tail_embed = vm["embed"](
        tvm.runtime.tensor(np.array(tail_ids, dtype="int32"), device), params
    ).numpy()

    patch_size = int(model_config["model_config"]["vision_config"].get("patch_size", 16))
    image_tensor = image_data.image if image_data.image.device == device else image_data.image.copyto(device)
    image_embed = vm["image_embed"](
        image_tensor,
        ShapeTuple([grid[1] * patch_size]),
        ShapeTuple([grid[2] * patch_size]),
        ShapeTuple([grid[1]]),
        ShapeTuple([grid[2]]),
        params,
    ).numpy()

    full_embed = np.concatenate([prefix_embed, image_embed, tail_embed], axis=0).astype("float16")
    position_ids, mrope_delta = qwen35_position_ids(
        len(prefix_ids), grid, len(tail_ids), generated_len=0
    )
    decode_embed = vm["embed"](
        tvm.runtime.tensor(np.array([args.decode_token_id], dtype="int32"), device), params
    ).numpy().reshape(1, 1, -1).astype("float16")

    ctx = {
        "vm": vm,
        "params": params,
        "device": device,
        "model_config": model_config,
        "full_embed": full_embed,
        "position_ids": position_ids.astype("int32"),
        "mrope_delta": np.array([[mrope_delta]], dtype="int32"),
        "decode_embed": decode_embed,
        "image_embed_len": len(image_data),
        "prompt_tokens": full_embed.shape[0],
    }
    if args.reuse_input_tensors:
        ctx.update(
            {
                "full_embed_tensor": tvm.runtime.tensor(
                    full_embed.reshape(1, *full_embed.shape), device
                ),
                "position_ids_tensor": tvm.runtime.tensor(position_ids.astype("int32"), device),
                "mrope_delta_tensor": tvm.runtime.tensor(ctx["mrope_delta"], device),
                "decode_embed_tensor": tvm.runtime.tensor(decode_embed, device),
                "last_token_index_tensor": tvm.runtime.tensor(
                    np.array([full_embed.shape[0] - 1], dtype="int32"), device
                ),
            }
        )
    return ctx


def create_states(ctx):
    from tvm.runtime import ShapeTuple

    vm = ctx["vm"]
    cfg = ctx["model_config"]["model_config"]
    try:
        create_kv_cache = vm["create_flashinfer_paged_kv_cache"]
    except Exception:
        create_kv_cache = vm["create_tir_paged_kv_cache"]
    kv_cache = create_kv_cache(
        ShapeTuple([1]),
        ShapeTuple([cfg.get("context_window_size", 2048)]),
        ShapeTuple([cfg.get("prefill_chunk_size", 320)]),
        ShapeTuple([16]),
        ShapeTuple([0]),
    )
    rnn_state = vm["create_rnn_state"](ShapeTuple([1]), ShapeTuple([1]))
    add_sequence = __import__("tvm").get_global_func("vm.builtin.kv_state_add_sequence")
    add_sequence(kv_cache, 0)
    add_sequence(rnn_state, 0)
    return kv_cache, rnn_state


def linear_layer_indices(ctx):
    pattern = ctx["model_config"]["model_config"].get("layer_type_pattern", [])
    return [idx for idx, layer_type in enumerate(pattern) if layer_type == "linear_attention"]


def rnn_state_shapes(ctx):
    cfg = ctx["model_config"]["model_config"]
    n_vh = int(cfg["linear_num_value_heads"])
    n_kh = int(cfg["linear_num_key_heads"])
    key_dim = int(cfg["linear_key_head_dim"])
    value_dim = int(cfg["linear_value_head_dim"])
    conv_kernel = int(cfg["linear_conv_kernel_dim"])
    qkv_dim = n_kh * key_dim * 2 + n_vh * value_dim
    return {
        0: ((1, n_vh, key_dim, value_dim), "float32"),
        1: ((1, conv_kernel - 1, qkv_dim), "float16"),
    }


def run_prefill_once(ctx):
    import tvm
    from tvm.runtime import ShapeTuple

    vm = ctx["vm"]
    device = ctx["device"]
    kv_cache, rnn_state = create_states(ctx)
    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    full_embed = ctx["full_embed"]
    seq_len = full_embed.shape[0]
    for state in (kv_cache, rnn_state):
        begin(state, ShapeTuple([0]), ShapeTuple([seq_len]))
    out = vm["batch_prefill_mrope"](
        ctx.get(
            "full_embed_tensor",
            tvm.runtime.tensor(full_embed.reshape(1, *full_embed.shape), device),
        ),
        ctx.get("position_ids_tensor", tvm.runtime.tensor(ctx["position_ids"], device)),
        ctx.get("mrope_delta_tensor", tvm.runtime.tensor(ctx["mrope_delta"], device)),
        ctx.get(
            "last_token_index_tensor",
            tvm.runtime.tensor(np.array([seq_len - 1], dtype="int32"), device),
        ),
        kv_cache,
        rnn_state,
        ctx["params"],
    )
    for state in (kv_cache, rnn_state):
        end(state)
    return out, kv_cache, rnn_state


def run_decode_once(ctx, kv_cache, rnn_state):
    import tvm
    from tvm.runtime import ShapeTuple

    vm = ctx["vm"]
    device = ctx["device"]
    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    for state in (kv_cache, rnn_state):
        begin(state, ShapeTuple([0]), ShapeTuple([1]))
    out = vm["decode_mrope"](
        ctx.get("decode_embed_tensor", tvm.runtime.tensor(ctx["decode_embed"], device)),
        ctx.get("mrope_delta_tensor", tvm.runtime.tensor(ctx["mrope_delta"], device)),
        kv_cache,
        rnn_state,
        ctx["params"],
    )
    for state in (kv_cache, rnn_state):
        end(state)
    return out


def run_decode_hidden_once(ctx, kv_cache, rnn_state):
    import tvm
    from tvm.runtime import ShapeTuple

    vm = ctx["vm"]
    device = ctx["device"]
    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    for state in (kv_cache, rnn_state):
        begin(state, ShapeTuple([0]), ShapeTuple([1]))
    out = vm["decode_mrope_to_last_hidden_states"](
        ctx.get("decode_embed_tensor", tvm.runtime.tensor(ctx["decode_embed"], device)),
        ctx.get("mrope_delta_tensor", tvm.runtime.tensor(ctx["mrope_delta"], device)),
        kv_cache,
        rnn_state,
        ctx["params"],
    )
    for state in (kv_cache, rnn_state):
        end(state)
    return out


def get_lm_head_func(ctx):
    vm = ctx["vm"]
    try:
        return "get_logits", vm["get_logits"]
    except AttributeError:
        return "get_token_ids", vm["get_token_ids"]


def run_lm_head_once(ctx, hidden_states, lm_head_func=None):
    if lm_head_func is None:
        _, lm_head_func = get_lm_head_func(ctx)
    return lm_head_func(hidden_states, ctx["params"])


def run_decode_state_once(ctx, kv_cache, rnn_state):
    import tvm
    from tvm.runtime import ShapeTuple

    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    for state in (kv_cache, rnn_state):
        begin(state, ShapeTuple([0]), ShapeTuple([1]))
    for state in (kv_cache, rnn_state):
        end(state)


def run_rnn_state_copy_once(ctx, rnn_state, buffers):
    import tvm
    from tvm.runtime import ShapeTuple

    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    get_state = tvm.get_global_func("vm.builtin.rnn_state_get")
    set_state = tvm.get_global_func("vm.builtin.rnn_state_set")
    begin(rnn_state, ShapeTuple([0]), ShapeTuple([1]))
    next_state = rnn_state
    for linear_idx in range(len(linear_layer_indices(ctx))):
        state_matrix, conv_state = buffers[linear_idx]
        get_state(next_state, linear_idx, 0, state_matrix)
        get_state(next_state, linear_idx, 1, conv_state)
        next_state = set_state(next_state, linear_idx, 0, state_matrix)
        next_state = set_state(next_state, linear_idx, 1, conv_state)
    end(next_state)
    return next_state


def run_rnn_raw_access_once(ctx, rnn_state):
    import tvm
    from tvm.runtime import ShapeTuple

    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    get_storage = tvm.get_global_func("vm.builtin.rnn_state_get_storage")
    get_seq = tvm.get_global_func("vm.builtin.rnn_state_get_seq_slot_ids")
    get_history = tvm.get_global_func("vm.builtin.rnn_state_get_history_slot_ids")
    begin(rnn_state, ShapeTuple([0]), ShapeTuple([1]))
    get_seq(rnn_state)
    get_history(rnn_state)
    for linear_idx in range(len(linear_layer_indices(ctx))):
        get_storage(rnn_state, linear_idx, 0)
        get_storage(rnn_state, linear_idx, 1)
    end(rnn_state)


def begin_decode_window(kv_cache, rnn_state):
    import tvm
    from tvm.runtime import ShapeTuple

    begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    for state in (kv_cache, rnn_state):
        begin(state, ShapeTuple([0]), ShapeTuple([1]))


def end_decode_window(kv_cache, rnn_state):
    import tvm

    end = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    for state in (kv_cache, rnn_state):
        end(state)


def run_decode_body_once(ctx, kv_cache, rnn_state):
    import tvm

    vm = ctx["vm"]
    device = ctx["device"]
    return vm["decode_mrope"](
        ctx.get("decode_embed_tensor", tvm.runtime.tensor(ctx["decode_embed"], device)),
        ctx.get("mrope_delta_tensor", tvm.runtime.tensor(ctx["mrope_delta"], device)),
        kv_cache,
        rnn_state,
        ctx["params"],
    )


def run_decode_hidden_body_once(ctx, kv_cache, rnn_state):
    import tvm

    vm = ctx["vm"]
    device = ctx["device"]
    return vm["decode_mrope_to_last_hidden_states"](
        ctx.get("decode_embed_tensor", tvm.runtime.tensor(ctx["decode_embed"], device)),
        ctx.get("mrope_delta_tensor", tvm.runtime.tensor(ctx["mrope_delta"], device)),
        kv_cache,
        rnn_state,
        ctx["params"],
    )


def sync(ctx) -> None:
    import tvm

    tvm.runtime.device("cuda", 0).sync()


def cuda_profiler_start(enabled: bool) -> None:
    if enabled:
        ctypes.CDLL("libcudart.so").cudaProfilerStart()


def cuda_profiler_stop(enabled: bool) -> None:
    if enabled:
        ctypes.CDLL("libcudart.so").cudaProfilerStop()


def main() -> None:
    configure_repo_env()
    args = parse_args()
    ctx = load_inputs(args)
    print(
        f"phase={args.phase} runs={args.runs} warmup_runs={args.warmup_runs} "
        f"prompt_tokens={ctx['prompt_tokens']} image_embed={ctx['image_embed_len']}"
    )

    if args.phase == "prefill":
        for _ in range(args.warmup_runs):
            run_prefill_once(ctx)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_prefill_once(ctx)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        print(f"prefill_seconds={elapsed:.6f} prefill_runs_per_second={args.runs / elapsed:.3f}")
        return

    _, kv_cache, rnn_state = run_prefill_once(ctx)
    if args.phase == "rnn-state-copy":
        import tvm

        shapes = rnn_state_shapes(ctx)
        buffers = []
        for _ in linear_layer_indices(ctx):
            state_matrix_shape, state_matrix_dtype = shapes[0]
            conv_shape, conv_dtype = shapes[1]
            buffers.append(
                (
                    tvm.runtime.empty(state_matrix_shape, state_matrix_dtype, ctx["device"]),
                    tvm.runtime.empty(conv_shape, conv_dtype, ctx["device"]),
                )
            )
        for _ in range(args.warmup_runs):
            rnn_state = run_rnn_state_copy_once(ctx, rnn_state, buffers)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            rnn_state = run_rnn_state_copy_once(ctx, rnn_state, buffers)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        print(
            f"rnn_state_copy_seconds={elapsed:.6f} "
            f"rnn_state_copy_runs_per_second={args.runs / elapsed:.3f}"
        )
        return

    if args.phase == "rnn-raw-access":
        for _ in range(args.warmup_runs):
            run_rnn_raw_access_once(ctx, rnn_state)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_rnn_raw_access_once(ctx, rnn_state)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        print(
            f"rnn_raw_access_seconds={elapsed:.6f} "
            f"rnn_raw_access_runs_per_second={args.runs / elapsed:.3f}"
        )
        return

    if args.phase == "decode-state":
        for _ in range(args.warmup_runs):
            run_decode_state_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_decode_state_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        print(
            f"decode_state_seconds={elapsed:.6f} "
            f"decode_state_runs_per_second={args.runs / elapsed:.3f}"
        )
        return

    if args.phase == "decode-body":
        begin_decode_window(kv_cache, rnn_state)
        for _ in range(args.warmup_runs):
            run_decode_body_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_decode_body_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        end_decode_window(kv_cache, rnn_state)
        print(
            f"decode_body_seconds={elapsed:.6f} "
            f"decode_body_tokens_per_second={args.runs / elapsed:.3f}"
        )
        return

    if args.phase == "decode-hidden-body":
        begin_decode_window(kv_cache, rnn_state)
        for _ in range(args.warmup_runs):
            run_decode_hidden_body_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_decode_hidden_body_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        end_decode_window(kv_cache, rnn_state)
        print(
            f"decode_hidden_body_seconds={elapsed:.6f} "
            f"decode_hidden_body_tokens_per_second={args.runs / elapsed:.3f}"
        )
        return

    if args.phase == "decode-hidden":
        for _ in range(args.warmup_runs):
            run_decode_hidden_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_decode_hidden_once(ctx, kv_cache, rnn_state)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        print(
            f"decode_hidden_seconds={elapsed:.6f} "
            f"decode_hidden_tokens_per_second={args.runs / elapsed:.3f}"
        )
        return

    if args.phase == "lm-head":
        hidden_states = run_decode_hidden_once(ctx, kv_cache, rnn_state)[0]
        lm_head_name, lm_head_func = get_lm_head_func(ctx)
        for _ in range(args.warmup_runs):
            run_lm_head_once(ctx, hidden_states, lm_head_func)
        sync(ctx)
        cuda_profiler_start(args.cuda_profiler_range)
        start = time.perf_counter()
        for _ in range(args.runs):
            run_lm_head_once(ctx, hidden_states, lm_head_func)
        sync(ctx)
        elapsed = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_range)
        print(
            f"lm_head_function={lm_head_name} "
            f"lm_head_seconds={elapsed:.6f} "
            f"lm_head_runs_per_second={args.runs / elapsed:.3f}"
        )
        return

    for _ in range(args.warmup_runs):
        run_decode_once(ctx, kv_cache, rnn_state)
    sync(ctx)
    cuda_profiler_start(args.cuda_profiler_range)
    start = time.perf_counter()
    for _ in range(args.runs):
        run_decode_once(ctx, kv_cache, rnn_state)
    sync(ctx)
    elapsed = time.perf_counter() - start
    cuda_profiler_stop(args.cuda_profiler_range)
    print(f"decode_seconds={elapsed:.6f} decode_tokens_per_second={args.runs / elapsed:.3f}")


if __name__ == "__main__":
    main()
