#!/usr/bin/env python3
"""Create a vLLM-loadable copy of the familiar-ai Qwen3.5 image checkpoint."""

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_HF_MODEL = "familiar-ai/logos-multitask-qwen3.5-2026-05-03-best"
DEFAULT_OUTPUT = "dist/logos-multitask-qwen3.5-2026-05-03-best-vllm-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def map_key(key: str) -> str:
    mappings = (
        (
            "model.language_model.language_model.language_model.",
            "language_model.model.",
        ),
        ("model.language_model.visual.", "visual."),
    )
    for old, new in mappings:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def copy_config_files(source_dir: Path, output_dir: Path) -> None:
    for path in source_dir.iterdir():
        if path.name in {
            "model.safetensors",
            "model.safetensors.index.json",
            ".gitattributes",
        }:
            continue
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def main() -> None:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    args = parse_args()
    output_dir = Path(args.output).expanduser()
    model_file = output_dir / "model.safetensors"
    if model_file.exists() and not args.force:
        print(f"Already exists: {model_file}")
        return

    source_dir = Path(snapshot_download(args.hf_model))
    source_model = source_dir / "model.safetensors"
    if not source_model.exists():
        raise FileNotFoundError(source_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config_files(source_dir, output_dir)

    tensors: dict[str, torch.Tensor] = {}
    remapped: dict[str, str] = {}
    with safe_open(str(source_model), framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for key in source.keys():
            new_key = map_key(key)
            if new_key in tensors:
                raise ValueError(f"Duplicate remapped key: {new_key}")
            tensors[new_key] = source.get_tensor(key)
            if new_key != key:
                remapped[key] = new_key

    save_file(tensors, str(model_file), metadata=metadata)
    weight_map = {key: "model.safetensors" for key in tensors}
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"source={source_dir}")
    print(f"output={output_dir}")
    print(f"tensors={len(tensors)} remapped={len(remapped)}")
    print(f"size_bytes={total_size}")


if __name__ == "__main__":
    main()
