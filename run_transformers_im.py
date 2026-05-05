#!/usr/bin/env python3
"""Run the Qwen3.5 image model through Transformers for MLC comparison."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


DEFAULT_HF_MODEL = "familiar-ai/logos-multitask-qwen3.5-2026-05-03-best"
PROCESSOR_USE_FAST_CHOICES = ("auto", "true", "false")


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
    parser.add_argument("--hf-model", default=os.environ.get("HF_MODEL", DEFAULT_HF_MODEL))
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--image", required=True)
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
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS/stop token IDs. Useful for fixed-length decode benchmarking.",
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
        "--profile-phases",
        action="store_true",
        help="Use a manual greedy loop to report prefill and decode timing separately.",
    )
    parser.add_argument(
        "--processor-use-fast",
        choices=PROCESSOR_USE_FAST_CHOICES,
        default=processor_use_fast_default(),
        help="Pass use_fast to AutoProcessor. 'auto' preserves Transformers' default.",
    )
    return parser.parse_args()


def load_image(
    path: str,
    fit_size: int,
    square_canvas: bool,
    fit_width: int = 0,
    fit_height: int = 0,
) -> Image.Image:
    from PIL import Image

    image = Image.open(Path(path).expanduser()).convert("RGB")
    if fit_size <= 0 and (fit_width <= 0 or fit_height <= 0):
        return image

    if fit_width > 0 and fit_height > 0:
        image.thumbnail((fit_width, fit_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (fit_width, fit_height), (255, 255, 255))
        canvas.paste(image, ((fit_width - image.width) // 2, (fit_height - image.height) // 2))
        return canvas

    image.thumbnail((fit_size, fit_size), Image.Resampling.LANCZOS)
    if not square_canvas:
        return image

    canvas = Image.new("RGB", (fit_size, fit_size), (255, 255, 255))
    canvas.paste(image, ((fit_size - image.width) // 2, (fit_size - image.height) // 2))
    return canvas


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    preprocess_start = time.perf_counter()
    fixed_image_size = args.fit_image_width > 0 or args.fit_image_height > 0
    if fixed_image_size and (args.fit_image_width <= 0 or args.fit_image_height <= 0):
        raise ValueError("Use --fit-image-width and --fit-image-height together.")
    if sum(bool(value) for value in (args.fit_image_size, args.fit_image_max_side, fixed_image_size)) > 1:
        raise ValueError(
            "Use only one of --fit-image-size, --fit-image-max-side, "
            "or --fit-image-width/--fit-image-height."
        )

    fit_size = args.fit_image_size or args.fit_image_max_side
    image = load_image(
        args.image,
        fit_size,
        square_canvas=args.fit_image_size > 0,
        fit_width=args.fit_image_width,
        fit_height=args.fit_image_height,
    )
    preprocess_seconds = time.perf_counter() - preprocess_start

    load_start = time.perf_counter()
    processor_kwargs = {"trust_remote_code": True}
    if args.processor_use_fast != "auto":
        processor_kwargs["use_fast"] = args.processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(args.hf_model, **processor_kwargs)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.hf_model,
        dtype=torch.float16,
        trust_remote_code=True,
        key_mapping={
            "model.language_model.language_model.language_model.": "model.language_model.",
            "model.language_model.visual.": "model.visual.",
        },
    ).to("cuda").eval()
    model_load_seconds = time.perf_counter() - load_start

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

    inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
    benchmark_runs = max(1, args.benchmark_runs)
    warmup_runs = max(0, args.warmup_runs)

    def run_generate():
        generate_kwargs = {}
        if args.ignore_eos:
            generate_kwargs["eos_token_id"] = None
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                **generate_kwargs,
            )

    def run_generate_profile():
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        generated = []
        prefill_seconds = 0.0
        decode_seconds = 0.0
        past_key_values = None

        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            first_inputs = model.prepare_inputs_for_generation(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                pixel_values=inputs.get("pixel_values"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                is_first_iteration=True,
            )
            outputs = model(**first_inputs)
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - start

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            past_key_values = outputs.past_key_values

            for _ in range(max(0, args.max_tokens - 1)):
                current_input_ids = torch.cat([input_ids, *generated], dim=1)
                if attention_mask is None:
                    current_attention_mask = torch.ones_like(current_input_ids)
                else:
                    current_attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], len(generated)),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )
                model_inputs = model.prepare_inputs_for_generation(
                    current_input_ids,
                    past_key_values=past_key_values,
                    attention_mask=current_attention_mask,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model(**model_inputs)
                torch.cuda.synchronize()
                decode_seconds += time.perf_counter() - start
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_token)
                past_key_values = outputs.past_key_values

        return torch.cat(generated, dim=1), prefill_seconds, decode_seconds

    for _ in range(warmup_runs):
        if args.profile_phases:
            _ = run_generate_profile()
        else:
            _ = run_generate()
        torch.cuda.synchronize()

    times = []
    output_ids = None
    new_tokens = None
    prefill_times = []
    decode_times = []
    for _ in range(benchmark_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.profile_phases:
            new_tokens, prefill_seconds, decode_seconds = run_generate_profile()
            prefill_times.append(prefill_seconds)
            decode_times.append(decode_seconds)
        else:
            output_ids = run_generate()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    if new_tokens is None:
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    text_out = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print(text_out)
    if benchmark_runs > 1 or warmup_runs > 0:
        generated_tokens = int(new_tokens.shape[1])
        total_seconds = sum(times)
        print(
            "benchmark "
            f"warmup_runs={warmup_runs} measured_runs={benchmark_runs} "
            f"generated_tokens_per_run={generated_tokens} "
            f"preprocess_seconds={preprocess_seconds:.6f} "
            f"model_load_seconds={model_load_seconds:.6f} "
            f"total_generate_seconds={total_seconds:.6f} "
            f"avg_generate_seconds={total_seconds / benchmark_runs:.6f} "
            f"generated_tokens_per_second={(generated_tokens * benchmark_runs) / total_seconds:.3f}"
        )
        if args.profile_phases:
            prefill_seconds = sum(prefill_times)
            decode_seconds = sum(decode_times)
            prompt_tokens = int(inputs["input_ids"].shape[1]) * benchmark_runs
            decode_tokens = max(0, generated_tokens - 1) * benchmark_runs
            print(
                "transformers_phase_metrics "
                "mode=manual_greedy "
                f"prefill_tokens={prompt_tokens} "
                f"prefill_seconds={prefill_seconds:.6f} "
                f"prefill_tokens_per_second="
                f"{prompt_tokens / prefill_seconds if prefill_seconds else 0.0:.3f} "
                f"decode_tokens={decode_tokens} "
                f"decode_seconds={decode_seconds:.6f} "
                f"decode_tokens_per_second="
                f"{decode_tokens / decode_seconds if decode_seconds else 0.0:.3f}"
            )


if __name__ == "__main__":
    main()
