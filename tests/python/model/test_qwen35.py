import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

tvm = pytest.importorskip("tvm")  # noqa: F401
pytest.importorskip("tvm_ffi")

from mlc_llm.interface.gen_config import (  # noqa: E402
    apply_qwen35_generation_defaults,
    apply_qwen35_preprocessor_config,
)
from mlc_llm.model.qwen35.qwen35_model import Qwen35Config, Qwen35LMHeadModel  # noqa: E402
from mlc_llm.serve.data import ImageData  # noqa: E402

pytestmark = [pytest.mark.unittest]


def _qwen35_0_8b_config_dict():
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
    return {
        "model_type": "qwen3_5",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "attention_bias": False,
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "layer_types": layer_types,
            "eos_token_id": 248044,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "vocab_size": 248320,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "depth": 12,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 768,
            "in_channels": 3,
            "intermediate_size": 3072,
            "model_type": "qwen3_5",
            "num_heads": 12,
            "num_position_embeddings": 2304,
            "out_hidden_size": 1024,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def test_qwen35_config_preserves_top_level_model_and_text_layer_types():
    cfg = Qwen35Config.from_dict(_qwen35_0_8b_config_dict())

    assert cfg.model_type == "qwen3_5"
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.vocab_size == 248320
    assert cfg.layer_types()[:4] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert cfg.num_linear_layers == 18
    assert cfg.num_attention_layers == 6
    assert cfg.mrope_section == (11, 11, 10)
    assert cfg.mrope_interleaved is True
    assert cfg.vision_config["hidden_size"] == 768


def test_qwen35_hides_decode_mrope_query_position_diagnostic_by_default():
    cfg = Qwen35Config.from_dict(_qwen35_0_8b_config_dict())
    model = Qwen35LMHeadModel(cfg)

    spec = model.get_default_spec()

    assert "decode_mrope_with_query_positions" not in spec.method_names


def test_qwen35_exports_decode_mrope_query_position_diagnostic_when_enabled():
    old_value = os.environ.get("MLC_QWEN35_EXPORT_DECODE_MROPE_QUERY_POSITIONS")
    os.environ["MLC_QWEN35_EXPORT_DECODE_MROPE_QUERY_POSITIONS"] = "1"
    try:
        cfg = Qwen35Config.from_dict(_qwen35_0_8b_config_dict())
        model = Qwen35LMHeadModel(cfg)
        spec = model.get_default_spec()
    finally:
        if old_value is None:
            os.environ.pop("MLC_QWEN35_EXPORT_DECODE_MROPE_QUERY_POSITIONS", None)
        else:
            os.environ["MLC_QWEN35_EXPORT_DECODE_MROPE_QUERY_POSITIONS"] = old_value

    assert "decode_mrope_with_query_positions" in spec.method_names
    method_spec = spec.method_specs[spec.method_names.index("decode_mrope_with_query_positions")]
    query_positions_spec = method_spec.arg_specs[
        method_spec.arg_names.index("query_positions")
    ]
    assert query_positions_spec.shape == [1]


def test_qwen35_omit_verify_export_keeps_serving_path_methods():
    old_value = os.environ.get("MLC_QWEN35_OMIT_VERIFY")
    os.environ["MLC_QWEN35_OMIT_VERIFY"] = "1"
    try:
        cfg = Qwen35Config.from_dict(_qwen35_0_8b_config_dict())
        model = Qwen35LMHeadModel(cfg)
        spec = model.get_default_spec()
    finally:
        if old_value is None:
            os.environ.pop("MLC_QWEN35_OMIT_VERIFY", None)
        else:
            os.environ["MLC_QWEN35_OMIT_VERIFY"] = old_value

    assert "batch_verify" not in spec.method_names
    assert "batch_verify_mrope" not in spec.method_names
    assert "image_embed" in spec.method_names
    assert "batch_prefill_mrope" in spec.method_names
    assert "decode_mrope" in spec.method_names
    assert "create_paged_kv_cache" in spec.method_names
    assert "create_rnn_state" in spec.method_names


def test_qwen35_preprocessor_config_updates_vision_resize_limits(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
            }
        ),
        encoding="utf-8",
    )
    cfg = Qwen35Config.from_dict(_qwen35_0_8b_config_dict())

    apply_qwen35_preprocessor_config(config_path, cfg)

    assert cfg.vision_config["min_pixels"] == 65536
    assert cfg.vision_config["max_pixels"] == 16777216
    assert cfg.vision_config["patch_size"] == 16
    assert cfg.vision_config["temporal_patch_size"] == 2
    assert cfg.vision_config["spatial_merge_size"] == 2


def test_qwen35_generation_defaults_use_nested_text_and_tokenizer_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_qwen35_0_8b_config_dict()), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "bos_token": None,
                "added_tokens_decoder": {
                    "248044": {"content": "<|endoftext|>"},
                    "248046": {"content": "<|im_end|>"},
                },
            }
        ),
        encoding="utf-8",
    )
    mlc_chat_config = SimpleNamespace(pad_token_id=None, bos_token_id=None, eos_token_id=None)

    apply_qwen35_generation_defaults(config_path, mlc_chat_config)

    assert mlc_chat_config.pad_token_id == 248044
    assert mlc_chat_config.bos_token_id is None
    assert mlc_chat_config.eos_token_id == 248044


def test_qwen35_image_embed_size_uses_dynamic_grid_and_python_rounding():
    config = {
        "model_type": "qwen3_5",
        "model_config": {
            "vision_config": {
                "patch_size": 16,
                "spatial_merge_size": 2,
                "min_pixels": 65536,
                "max_pixels": 16777216,
            }
        },
    }

    embed_size, grid_thw = ImageData.get_embed_size(config, (80, 80), return_grid=True)

    # Python/HF round(80 / 32) is 2, then min_pixels upsizes the image to 256x256.
    assert grid_thw == (1, 16, 16)
    assert embed_size == 64


def test_qwen35_image_embed_size_rejects_extreme_aspect_ratio():
    config = {
        "model_type": "qwen3_5",
        "model_config": {
            "vision_config": {
                "patch_size": 16,
                "spatial_merge_size": 2,
                "min_pixels": 65536,
                "max_pixels": 16777216,
            }
        },
    }

    with pytest.raises(ValueError, match="aspect ratio"):
        ImageData.get_embed_size(config, (1, 256), return_grid=True)
