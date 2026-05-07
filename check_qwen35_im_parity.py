#!/usr/bin/env python3
"""Check Qwen3.5 image generation parity between Transformers and MLC."""

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

from run_im import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_LIB,
    configure_repo_env,
    image_file_to_url,
    qwen35_assistant_prefix,
    qwen35_user_prefix,
)
from run_transformers_im import (
    DEFAULT_HF_MODEL,
    PROCESSOR_USE_FAST_CHOICES,
    load_image,
    processor_use_fast_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default=os.environ.get("HF_MODEL", DEFAULT_HF_MODEL))
    parser.add_argument("--model", default=os.environ.get("MLC_MODEL", str(DEFAULT_MODEL)))
    parser.add_argument("--model-lib", default=os.environ.get("MLC_MODEL_LIB", str(DEFAULT_MODEL_LIB)))
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--fit-image-width", type=int, default=640)
    parser.add_argument("--fit-image-height", type=int, default=480)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--processor-use-fast",
        choices=PROCESSOR_USE_FAST_CHOICES,
        default=processor_use_fast_default(),
        help="Pass use_fast to AutoProcessor. 'auto' preserves Transformers' default.",
    )
    parser.add_argument("--min-token-prefix", type=int, default=0)
    parser.add_argument("--min-text-ratio", type=float, default=0.40)
    parser.add_argument(
        "--required-substring",
        action="append",
        default=[],
        help="Substring that must appear in both decoded outputs. May be repeated.",
    )
    return parser.parse_args()


def common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def run_transformers(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    image = load_image(
        args.image,
        fit_size=0,
        square_canvas=False,
        fit_width=args.fit_image_width,
        fit_height=args.fit_image_height,
    )
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
    inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0], new_tokens[0].tolist()


def run_mlc(args: argparse.Namespace):
    configure_repo_env()

    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve import data
    from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine

    model = Path(args.model).expanduser()
    model_lib = Path(args.model_lib).expanduser()
    with (model / "mlc-chat-config.json").open(encoding="utf-8") as file:
        model_config = json.load(file)
    image_url = image_file_to_url(
        args.image,
        fit_width=args.fit_image_width,
        fit_height=args.fit_image_height,
    )
    prompt = [
        data.TextData(qwen35_user_prefix()),
        data.TextData("<|vision_start|>"),
        data.ImageData.from_url(image_url, model_config),
        data.TextData(f"<|vision_end|>{args.prompt}{qwen35_assistant_prefix()}"),
    ]
    engine = SyncMLCEngine(
        model=str(model),
        model_lib=str(model_lib),
        mode="server",
        engine_config=EngineConfig(max_total_sequence_length=2048, max_num_sequence=1),
    )
    outputs, _ = engine.generate(
        [prompt],
        GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop_token_ids=[248046, 248044],
        ),
    )
    text = outputs[0][0]
    token_ids = engine.tokenizer.encode(text)
    engine.reset()
    return text, token_ids


def main() -> None:
    args = parse_args()
    hf_text, hf_ids = run_transformers(args)
    mlc_text, mlc_ids = run_mlc(args)
    prefix_len = common_prefix_len(hf_ids, mlc_ids)
    text_ratio = difflib.SequenceMatcher(None, hf_text, mlc_text).ratio()

    print(f"transformers_text={hf_text!r}")
    print(f"mlc_text={mlc_text!r}")
    print(
        "parity "
        f"hf_tokens={len(hf_ids)} mlc_tokens={len(mlc_ids)} "
        f"common_token_prefix={prefix_len} text_similarity={text_ratio:.3f}"
    )

    hf_text_lower = hf_text.lower()
    mlc_text_lower = mlc_text.lower()
    for substring in args.required_substring:
        substring_lower = substring.lower()
        if substring_lower not in hf_text_lower or substring_lower not in mlc_text_lower:
            raise SystemExit(
                "Parity check failed: required substring missing from one output: "
                f"{substring!r}"
            )

    if prefix_len < args.min_token_prefix and text_ratio < args.min_text_ratio:
        raise SystemExit(
            "Parity check failed: "
            f"common_token_prefix={prefix_len} < {args.min_token_prefix} and "
            f"text_similarity={text_ratio:.3f} < {args.min_text_ratio:.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
