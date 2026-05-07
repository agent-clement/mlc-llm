"""
Implementation for Qwen3.5 GatedDeltaNet hybrid architecture.
75% GatedDeltaNet (recurrent linear attention), 25% standard GQA softmax attention.
"""

import dataclasses
import math
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
from tvm import relax as R
from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

from mlc_llm import op as op_ext
from mlc_llm.model.vision import ImageProcessor
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.op.mrope import (
    MultimodalRotaryEmbedding,
    VisionPositionMetadata,
    apply_multimodal_rotary_pos_emb,
)
from mlc_llm.op import triton as triton_ops
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


def _is_static_one(value: Any) -> bool:
    return value == 1 or getattr(value, "value", None) == 1


@dataclasses.dataclass
class Qwen35Config(ConfigBase):
    """Configuration of the Qwen3.5 model."""

    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_hidden_layers: int = 0
    num_key_value_heads: int = 0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    rope_theta: int = 10000000
    head_dim: int = 256
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    model_type: str = "qwen3_5"
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vision_config: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006
    mrope_section: Optional[Tuple[int, int, int]] = None  # noqa: UP006
    mrope_interleaved: bool = True
    # GatedDeltaNet-specific
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    layer_type_pattern: Optional[List[str]] = None  # noqa: UP006
    partial_rotary_factor: float = 0.25
    # Runtime
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    dtype: str = "float32"
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        # Handle VLM wrapper: Qwen3.5 HF config has all text params inside text_config
        if "text_config" in self.kwargs:
            text_config = self.kwargs.pop("text_config")
            if isinstance(text_config, dict):
                field_names = {f.name for f in dataclasses.fields(self.__class__)}
                for k, v in text_config.items():
                    if k == "layer_types":
                        self.layer_type_pattern = v
                    elif k in field_names and k not in ["kwargs", "model_type"]:
                        setattr(self, k, v)
                    else:
                        self.kwargs[k] = v
                # Extract rope params from nested rope_parameters
                rope_params = text_config.get("rope_parameters", {})
                if isinstance(rope_params, dict):
                    if "rope_theta" in rope_params:
                        self.rope_theta = rope_params["rope_theta"]
                    if "partial_rotary_factor" in rope_params:
                        self.partial_rotary_factor = rope_params["partial_rotary_factor"]
                    if "mrope_section" in rope_params:
                        self.mrope_section = tuple(int(i) for i in rope_params["mrope_section"])
                    if "mrope_interleaved" in rope_params:
                        self.mrope_interleaved = bool(rope_params["mrope_interleaved"])

        if "vision_config" in self.kwargs:
            vision_config = self.kwargs.pop("vision_config")
            if isinstance(vision_config, dict):
                self.vision_config = vision_config
        if "layer_types" in self.kwargs:
            self.layer_type_pattern = self.kwargs.pop("layer_types")

        # Also handle rope_parameters at top level
        if "rope_parameters" in self.kwargs:
            rope_params = self.kwargs.pop("rope_parameters")
            if isinstance(rope_params, dict):
                if "rope_theta" in rope_params:
                    self.rope_theta = rope_params["rope_theta"]
                if "partial_rotary_factor" in rope_params:
                    self.partial_rotary_factor = rope_params["partial_rotary_factor"]
                if "mrope_section" in rope_params:
                    self.mrope_section = tuple(int(i) for i in rope_params["mrope_section"])
                if "mrope_interleaved" in rope_params:
                    self.mrope_interleaved = bool(rope_params["mrope_interleaved"])

        if self.mrope_section is not None and len(self.mrope_section) != 3:
            raise ValueError(f"mrope_section must contain 3 integers, got {self.mrope_section}.")
        if self.layer_type_pattern is not None:
            self.layer_type_pattern = [str(layer_type) for layer_type in self.layer_type_pattern]
            if len(self.layer_type_pattern) != self.num_hidden_layers:
                raise ValueError(
                    "layer_types length must equal num_hidden_layers, got "
                    f"{len(self.layer_type_pattern)} vs {self.num_hidden_layers}."
                )
            for layer_type in self.layer_type_pattern:
                if layer_type not in ["linear_attention", "full_attention"]:
                    raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}.")

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 2048)

    @property
    def num_linear_layers(self) -> int:
        """Number of GatedDeltaNet linear attention layers."""
        return self.layer_types().count("linear_attention")

    @property
    def num_attention_layers(self) -> int:
        """Number of full attention layers."""
        return self.layer_types().count("full_attention")

    def layer_types(self) -> List[str]:  # noqa: UP006
        """Returns list of layer types: 'linear_attention' or 'full_attention'."""
        if self.layer_type_pattern is not None:
            return list(self.layer_type_pattern)
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types

    @property
    def vision_metadata(self) -> VisionPositionMetadata:
        """Metadata required for Qwen3.5 multimodal position-id construction."""
        spatial_merge_size = int(self.vision_config.get("spatial_merge_size", 2))
        return VisionPositionMetadata(
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            spatial_merge_size=spatial_merge_size,
            tokens_per_second=float(self.vision_config.get("tokens_per_second", 1.0)),
        )


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "gelu_pytorch_tanh": partial(nn.gelu, approximate="tanh"),
    "relu": nn.relu,
    "silu": nn.silu,
}


class Qwen35Embedding(nn.Embedding):
    def lm_head_forward(self, x: nn.Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Qwen35VisionMLP(nn.Module):
    def __init__(self, vision_config: Dict[str, Any]):  # noqa: UP006
        hidden_size = int(vision_config["hidden_size"])
        intermediate_size = int(vision_config["intermediate_size"])
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = ACT2FN.get(str(vision_config.get("hidden_act", "gelu")), nn.gelu)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen35VisionPatchEmbed(nn.Module):
    def __init__(self, vision_config: Dict[str, Any]):  # noqa: UP006
        self.patch_size = int(vision_config.get("patch_size", 16))
        self.temporal_patch_size = int(vision_config.get("temporal_patch_size", 2))
        self.in_channels = int(vision_config.get("in_channels", 3))
        self.embed_dim = int(vision_config["hidden_size"])
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3D(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.proj(pixel_values)
        hidden_states = op.permute_dims(hidden_states, axes=(0, 2, 3, 4, 1))
        return op.reshape(hidden_states, (-1, self.embed_dim))


class Qwen35VisionAttention(nn.Module):
    def __init__(self, vision_config: Dict[str, Any]):  # noqa: UP006
        self.dim = int(vision_config["hidden_size"])
        self.num_heads = int(vision_config["num_heads"])
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.scaling = self.head_dim**-0.5
        self.merge = int(vision_config.get("spatial_merge_size", 2))
        rotary_dim = self.head_dim // 2
        self.inv_freq = 1.0 / (
            10000.0 ** (np.arange(0, rotary_dim, 2, dtype="float32") / np.float32(rotary_dim))
        )

    def _apply_vision_rope(self, q: Tensor, k: Tensor, grid_h: int, grid_w: int):
        head_dim = self.head_dim
        half_dim = head_dim // 2
        quarter_dim = head_dim // 4
        merge = self.merge
        inv_freq = nn.Tensor.from_const(self.inv_freq)

        def create_vision_rope_func(dtype):
            tx = 256

            @T.prim_func
            def vision_rope_func(
                q_in: T.handle,
                k_in: T.handle,
                inv: T.handle,
                q_out: T.handle,
                k_out: T.handle,
                image_grid_h: T.int64(),
                image_grid_w: T.int64(),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
                seq_len, num_heads = T.int64(), T.int64()
                q_buf = T.match_buffer(q_in, (seq_len, num_heads, head_dim), dtype=dtype)
                k_buf = T.match_buffer(k_in, (seq_len, num_heads, head_dim), dtype=dtype)
                inv_buf = T.match_buffer(inv, (quarter_dim,), dtype="float32")
                qo_buf = T.match_buffer(q_out, (seq_len, num_heads, head_dim), dtype=dtype)
                ko_buf = T.match_buffer(k_out, (seq_len, num_heads, head_dim), dtype=dtype)
                merged_w = image_grid_w // merge
                tokens_per_t = image_grid_h * image_grid_w
                total = seq_len * num_heads * T.int64(head_dim)
                for bx in T.thread_binding(0, T.ceildiv(total, T.int64(tx)), thread="blockIdx.x"):
                    for tx_idx in T.thread_binding(0, tx, thread="threadIdx.x"):
                        idx = bx * T.int64(tx) + T.cast(tx_idx, "int64")
                        if idx < total:
                            d = T.floormod(idx, T.int64(head_dim))
                            tmp = idx // T.int64(head_dim)
                            h = T.floormod(tmp, num_heads)
                            s = tmp // num_heads
                            with T.sblock("qwen35_vision_rope"):
                                vs = T.axis.spatial(seq_len, s)
                                vh = T.axis.spatial(num_heads, h)
                                vd = T.axis.spatial(T.int64(head_dim), d)
                                T.reads(q_buf[vs, vh, vd], k_buf[vs, vh, vd], inv_buf[0:quarter_dim])
                                T.writes(qo_buf[vs, vh, vd], ko_buf[vs, vh, vd])
                                rem = T.floormod(vs, tokens_per_t)
                                block = rem // T.int64(merge * merge)
                                intra = T.floormod(rem, T.int64(merge * merge))
                                bh = block // merged_w
                                bw = T.floormod(block, merged_w)
                                ih = intra // T.int64(merge)
                                iw = T.floormod(intra, T.int64(merge))
                                row = bh * T.int64(merge) + ih
                                col = bw * T.int64(merge) + iw
                                base_d = T.floormod(vd, T.int64(half_dim))
                                freq_idx = T.floormod(base_d, T.int64(quarter_dim))
                                pos = T.Select(base_d < T.int64(quarter_dim), row, col)
                                angle = T.cast(pos, "float32") * inv_buf[freq_idx]
                                cos = tirx.cos(angle)
                                sin = tirx.sin(angle)
                                rot_d = T.Select(
                                    vd < T.int64(half_dim),
                                    vd + T.int64(half_dim),
                                    vd - T.int64(half_dim),
                                )
                                q_rot = T.Select(
                                    vd < T.int64(half_dim),
                                    -T.cast(q_buf[vs, vh, rot_d], "float32"),
                                    T.cast(q_buf[vs, vh, rot_d], "float32"),
                                )
                                k_rot = T.Select(
                                    vd < T.int64(half_dim),
                                    -T.cast(k_buf[vs, vh, rot_d], "float32"),
                                    T.cast(k_buf[vs, vh, rot_d], "float32"),
                                )
                                qo_buf[vs, vh, vd] = T.cast(
                                    T.cast(q_buf[vs, vh, vd], "float32") * cos + q_rot * sin,
                                    dtype,
                                )
                                ko_buf[vs, vh, vd] = T.cast(
                                    T.cast(k_buf[vs, vh, vd], "float32") * cos + k_rot * sin,
                                    dtype,
                                )

            return vision_rope_func

        q_out, k_out = op.tensor_ir_op(
            create_vision_rope_func(q.dtype),
            "qwen35_vision_rope",
            [q, k, inv_freq, grid_h, grid_w],
            [Tensor.placeholder(q.shape, q.dtype), Tensor.placeholder(k.shape, k.dtype)],
        )
        return q_out, k_out

    def forward(self, hidden_states: Tensor, grid_h: int, grid_w: int) -> Tensor:
        seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = op.reshape(qkv, (seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = op.split(qkv, 3, axis=1)
        q = op.reshape(q, (seq_len, self.num_heads, self.head_dim))
        k = op.reshape(k, (seq_len, self.num_heads, self.head_dim))
        v = op.reshape(v, (seq_len, self.num_heads, self.head_dim))
        q, k = self._apply_vision_rope(q, k, grid_h, grid_w)
        q = op.permute_dims(q, axes=(1, 0, 2))
        k = op.permute_dims(k, axes=(1, 2, 0))
        v = op.permute_dims(v, axes=(1, 0, 2))
        attn_weights = op.matmul(q, k) * self.scaling
        attn_weights = op.softmax(attn_weights, axis=-1)
        attn_output = op.matmul(attn_weights, v)
        attn_output = op.permute_dims(attn_output, axes=(1, 0, 2))
        attn_output = op.reshape(attn_output, (seq_len, self.dim))
        return self.proj(attn_output)


class Qwen35VisionBlock(nn.Module):
    def __init__(self, vision_config: Dict[str, Any]):  # noqa: UP006
        hidden_size = int(vision_config["hidden_size"])
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen35VisionAttention(vision_config)
        self.mlp = Qwen35VisionMLP(vision_config)

    def forward(self, hidden_states: Tensor, grid_h: int, grid_w: int) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), grid_h, grid_w)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen35VisionPatchMerger(nn.Module):
    def __init__(self, vision_config: Dict[str, Any]):  # noqa: UP006
        hidden_size = int(vision_config["hidden_size"])
        merge = int(vision_config.get("spatial_merge_size", 2))
        self.hidden_size = hidden_size * merge * merge
        self.out_hidden_size = int(vision_config.get("out_hidden_size", hidden_size))
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, self.out_hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.merge = merge

    def forward(self, hidden_states: Tensor, grid_h: int, grid_w: int) -> Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = op.reshape(hidden_states, (-1, self.hidden_size))
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen35VisionModel(nn.Module):
    def __init__(self, config: Qwen35Config):
        vision_config = config.vision_config
        self.patch_size = int(vision_config.get("patch_size", 16))
        self.temporal_patch_size = int(vision_config.get("temporal_patch_size", 2))
        self.patch_embed = Qwen35VisionPatchEmbed(vision_config)
        self.pos_embed = nn.Embedding(
            int(vision_config.get("num_position_embeddings", 2304)),
            int(vision_config["hidden_size"]),
        )
        # The vision reorder/interpolation kernel consumes the full positional
        # table directly. It is small and not on the decode hot path, so keep it
        # dense even for weight-only quantized models.
        self.pos_embed.no_quantization = True
        self.merge = int(vision_config.get("spatial_merge_size", 2))
        self.hidden_size = int(vision_config["hidden_size"])
        self.num_grid_per_side = int(int(vision_config.get("num_position_embeddings", 2304)) ** 0.5)
        self.blocks = nn.ModuleList(
            [Qwen35VisionBlock(vision_config) for _ in range(int(vision_config["depth"]))]
        )
        self.merger = Qwen35VisionPatchMerger(vision_config)

    def _block_reorder_and_pos_embed(
        self,
        hidden_states: Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tensor:
        merge = self.merge
        hidden_size = self.hidden_size
        base_side = self.num_grid_per_side

        def create_reorder_func(dtype, pos_dtype):
            tx = 256

            @T.prim_func
            def reorder_func(
                hidden: T.handle,
                pos_weight: T.handle,
                out: T.handle,
                image_grid_h: T.int64(),
                image_grid_w: T.int64(),
            ):
                T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
                seq_len = T.int64()
                hidden_buf = T.match_buffer(hidden, (seq_len, hidden_size), dtype=dtype)
                pos_buf = T.match_buffer(
                    pos_weight, (base_side * base_side, hidden_size), dtype=pos_dtype
                )
                out_buf = T.match_buffer(out, (seq_len, hidden_size), dtype=dtype)
                grid_t = seq_len // (image_grid_h * image_grid_w)
                merged_h = image_grid_h // merge
                merged_w = image_grid_w // merge
                total = seq_len * T.int64(hidden_size)
                for bx in T.thread_binding(0, T.ceildiv(total, T.int64(tx)), thread="blockIdx.x"):
                    for tx_idx in T.thread_binding(0, tx, thread="threadIdx.x"):
                        idx = bx * T.int64(tx) + T.cast(tx_idx, "int64")
                        if idx < total:
                            c = T.floormod(idx, T.int64(hidden_size))
                            dst_idx = idx // T.int64(hidden_size)
                            intra = T.floormod(dst_idx, T.int64(merge * merge))
                            tile = dst_idx // T.int64(merge * merge)
                            iw = T.floormod(intra, T.int64(merge))
                            ih = intra // T.int64(merge)
                            bw = T.floormod(tile, merged_w)
                            tmp = tile // merged_w
                            bh = T.floormod(tmp, merged_h)
                            t = tmp // merged_h
                            with T.sblock("qwen35_reorder_pos"):
                                vc = T.axis.spatial(T.int64(hidden_size), c)
                                T.reads(hidden_buf[0:seq_len, vc], pos_buf[0 : base_side * base_side, vc])
                                T.writes(out_buf[dst_idx, vc])
                                row = bh * T.int64(merge) + ih
                                col = bw * T.int64(merge) + iw
                                src_idx = (t * image_grid_h + row) * image_grid_w + col
                                den_h = T.Select(image_grid_h > 1, image_grid_h - 1, T.int64(1))
                                den_w = T.Select(image_grid_w > 1, image_grid_w - 1, T.int64(1))
                                h_num = row * (base_side - 1)
                                w_num = col * (base_side - 1)
                                h_floor = h_num // den_h
                                w_floor = w_num // den_w
                                h_ceil = T.Select(
                                    h_floor + 1 < base_side,
                                    h_floor + 1,
                                    T.int64(base_side - 1),
                                )
                                w_ceil = T.Select(
                                    w_floor + 1 < base_side,
                                    w_floor + 1,
                                    T.int64(base_side - 1),
                                )
                                dh = h_num - h_floor * den_h
                                dw = w_num - w_floor * den_w
                                wh0 = T.cast(den_h - dh, "float32") / T.cast(den_h, "float32")
                                wh1 = T.cast(dh, "float32") / T.cast(den_h, "float32")
                                ww0 = T.cast(den_w - dw, "float32") / T.cast(den_w, "float32")
                                ww1 = T.cast(dw, "float32") / T.cast(den_w, "float32")
                                p00 = T.cast(
                                    pos_buf[h_floor * base_side + w_floor, vc], "float32"
                                )
                                p01 = T.cast(
                                    pos_buf[h_floor * base_side + w_ceil, vc], "float32"
                                )
                                p10 = T.cast(
                                    pos_buf[h_ceil * base_side + w_floor, vc], "float32"
                                )
                                p11 = T.cast(
                                    pos_buf[h_ceil * base_side + w_ceil, vc], "float32"
                                )
                                pos_val = (
                                    p00 * wh0 * ww0
                                    + p01 * wh0 * ww1
                                    + p10 * wh1 * ww0
                                    + p11 * wh1 * ww1
                                )
                                out_buf[dst_idx, vc] = hidden_buf[src_idx, vc] + T.cast(
                                    pos_val, dtype
                                )

            return reorder_func

        seq_len, _ = hidden_states.shape
        return op.tensor_ir_op(
            create_reorder_func(hidden_states.dtype, self.pos_embed.weight.dtype),
            "qwen35_reorder_pos",
            [hidden_states, self.pos_embed.weight, grid_h, grid_w],
            [Tensor.placeholder([seq_len, hidden_size], hidden_states.dtype)],
        )

    def forward(self, pixel_values: Tensor, grid_h: int, grid_w: int) -> Tensor:
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = self._block_reorder_and_pos_embed(hidden_states, grid_h, grid_w)
        for block in self.blocks:
            hidden_states = block(hidden_states, grid_h, grid_w)
        return self.merger(hidden_states, grid_h, grid_w)


class Qwen35Attention(nn.Module):
    """Standard GQA attention with output gate for full_attention layers (every 4th layer).

    attn_output_gate=True: q_proj outputs 2*num_heads*head_dim, split into (Q, gate).
    Gate is sigmoid-applied to attention output before o_proj.
    """

    def __init__(self, config: Qwen35Config):
        self.head_dim = config.head_dim
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.rope_theta = config.rope_theta
        self.mrope_section = config.mrope_section
        self.mrope_interleaved = config.mrope_interleaved

        # c_attn: Q (2x for gate) + K + V fused projection
        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=(2 * self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)
        self.k_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)

    def _apply_mrope(
        self,
        q: Tensor,
        k: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],  # noqa: UP006
    ) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        if self.mrope_section is None:
            return q, k
        rotary_dim = self.rotary_dim
        if rotary_dim == self.head_dim:
            return apply_multimodal_rotary_pos_emb(
                q,
                k,
                position_embeddings[0],
                position_embeddings[1],
                self.mrope_section,
                interleaved=self.mrope_interleaved,
            )
        q_rot, q_pass = op.split(q, [rotary_dim], axis=-1)
        k_rot, k_pass = op.split(k, [rotary_dim], axis=-1)
        q_rot, k_rot = apply_multimodal_rotary_pos_emb(
            q_rot,
            k_rot,
            position_embeddings[0],
            position_embeddings[1],
            self.mrope_section,
            interleaved=self.mrope_interleaved,
        )
        return op.concat([q_rot, q_pass], dim=-1), op.concat([k_rot, k_pass], dim=-1)

    def _apply_mrope_q(
        self,
        q: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],  # noqa: UP006
    ) -> Tensor:
        if self.mrope_section is None:
            return q
        rotary_dim = self.rotary_dim
        if rotary_dim == self.head_dim:
            q_embed, _ = apply_multimodal_rotary_pos_emb(
                q,
                q,
                position_embeddings[0],
                position_embeddings[1],
                self.mrope_section,
                interleaved=self.mrope_interleaved,
            )
            return q_embed
        q_rot, q_pass = op.split(q, [rotary_dim], axis=-1)
        q_rot, _ = apply_multimodal_rotary_pos_emb(
            q_rot,
            q_rot,
            position_embeddings[0],
            position_embeddings[1],
            self.mrope_section,
            interleaved=self.mrope_interleaved,
        )
        return op.concat([q_rot, q_pass], dim=-1)

    def _apply_mrope_k(
        self,
        k: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],  # noqa: UP006
    ) -> Tensor:
        if self.mrope_section is None:
            return k
        rotary_dim = self.rotary_dim
        if rotary_dim == self.head_dim:
            _, k_embed = apply_multimodal_rotary_pos_emb(
                k,
                k,
                position_embeddings[0],
                position_embeddings[1],
                self.mrope_section,
                interleaved=self.mrope_interleaved,
            )
            return k_embed
        k_rot, k_pass = op.split(k, [rotary_dim], axis=-1)
        _, k_rot = apply_multimodal_rotary_pos_emb(
            k_rot,
            k_rot,
            position_embeddings[0],
            position_embeddings[1],
            self.mrope_section,
            interleaved=self.mrope_interleaved,
        )
        return op.concat([k_rot, k_pass], dim=-1)

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # noqa: UP006
    ):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        # c_attn outputs flat: [Q_with_gate (h_q * 2 * d), K (h_kv * d), V (h_kv * d)]
        proj = self.c_attn(hidden_states)
        # Reshape to heads: (b, s, 2*h_q + 2*h_kv, d)
        proj = op.reshape(proj, (b, s, 2 * h_q + 2 * h_kv, d))
        # Split: first 2*h_q heads have interleaved [Q, gate] per head, then h_kv K, h_kv V
        q_gate, k, v = op.split(proj, [2 * h_q, 2 * h_q + h_kv], axis=2)
        # q_gate shape: (b, s, 2*h_q, d). Even heads are Q, odd heads are gate
        # But HF layout is per-head [Q_d, gate_d], so reshape to (b, s, h_q, 2*d) then split
        q_gate = op.reshape(q_gate, (b, s, h_q, 2 * d))
        q, gate = op.split(q_gate, [d], axis=3)
        # gate: (b, s, h_q, d) -> flatten to (b, s, h_q*d)
        gate = op.reshape(gate, (b, s, h_q * d))
        use_separate_decode_attention = (
            os.environ.get("MLC_QWEN35_SEPARATE_QKV_ATTENTION", "0") == "1"
            and _is_static_one(s)
        )
        use_combined_kv_cross_attention = (
            os.environ.get("MLC_QWEN35_COMBINED_KV_CROSS_ATTENTION", "0") == "1"
            and _is_static_one(s)
        )
        use_fa2_paged_decode_attention = (
            os.environ.get("MLC_QWEN35_FA2_PAGED_DECODE_ATTENTION", "0") == "1"
            and _is_static_one(s)
        )
        use_explicit_paged_cross_attention = (
            os.environ.get("MLC_QWEN35_EXPLICIT_PAGED_CROSS_ATTENTION", "0") == "1"
            and _is_static_one(s)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        if position_embeddings is not None:
            q, k = self._apply_mrope(q, k, position_embeddings)
            if use_fa2_paged_decode_attention:
                paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id, k, v)
                output, _ = paged_kv_cache.fa2_paged_decode_attention(
                    layer_id,
                    q,
                    self.head_dim,
                    sm_scale=self.head_dim**-0.5,
                )
                output = op.reshape(output, (b, s, h_q * d))
                output = output * op.sigmoid(gate)
                return self.o_proj(output)
            if use_explicit_paged_cross_attention:
                paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id, k, v)
                output, _ = paged_kv_cache.cross_attention_with_paged_metadata(
                    layer_id,
                    q,
                    self.head_dim,
                    sm_scale=self.head_dim**-0.5,
                )
                output = op.reshape(output, (b, s, h_q * d))
                output = output * op.sigmoid(gate)
                return self.o_proj(output)
            if use_combined_kv_cross_attention:
                output, _ = paged_kv_cache.append_mha_kv_cross_attention(
                    layer_id,
                    q,
                    k,
                    v,
                    self.head_dim,
                    sm_scale=self.head_dim**-0.5,
                )
                output = op.reshape(output, (b, s, h_q * d))
                output = output * op.sigmoid(gate)
                return self.o_proj(output)
            if use_separate_decode_attention:
                paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id, k, v)
                output, _ = paged_kv_cache.cross_attention(
                    layer_id,
                    q,
                    self.head_dim,
                    sm_scale=self.head_dim**-0.5,
                )
                output = op.reshape(output, (b, s, h_q * d))
                output = output * op.sigmoid(gate)
                return self.o_proj(output)
            qkv = op.concat([q, k, v], dim=2)
            output = op.reshape(
                paged_kv_cache.attention_with_fused_qkv(
                    layer_id, qkv, self.num_attention_heads, sm_scale=self.head_dim**-0.5
                ),
                (b, s, h_q * d),
            )
            output = output * op.sigmoid(gate)
            return self.o_proj(output)
        if use_fa2_paged_decode_attention:
            paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id, k, v)
            output, _ = paged_kv_cache.fa2_paged_decode_attention(
                layer_id,
                q,
                self.head_dim,
                sm_scale=self.head_dim**-0.5,
            )
            output = op.reshape(output, (b, s, h_q * d))
            output = output * op.sigmoid(gate)
            return self.o_proj(output)
        if use_explicit_paged_cross_attention:
            paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id, k, v)
            output, _ = paged_kv_cache.cross_attention_with_paged_metadata(
                layer_id,
                q,
                self.head_dim,
                sm_scale=self.head_dim**-0.5,
            )
            output = op.reshape(output, (b, s, h_q * d))
            output = output * op.sigmoid(gate)
            return self.o_proj(output)
        if use_combined_kv_cross_attention:
            output, _ = paged_kv_cache.append_mha_kv_cross_attention(
                layer_id,
                q,
                k,
                v,
                self.head_dim,
                sm_scale=self.head_dim**-0.5,
            )
            output = op.reshape(output, (b, s, h_q * d))
            output = output * op.sigmoid(gate)
            return self.o_proj(output)
        if use_separate_decode_attention:
            paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id, k, v)
            output, _ = paged_kv_cache.cross_attention(
                layer_id,
                q,
                self.head_dim,
                sm_scale=self.head_dim**-0.5,
            )
            output = op.reshape(output, (b, s, h_q * d))
            output = output * op.sigmoid(gate)
            return self.o_proj(output)
        qkv = op.concat([q, k, v], dim=2)
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(
                layer_id, qkv, self.num_attention_heads, sm_scale=self.head_dim**-0.5
            ),
            (b, s, h_q * d),
        )
        # Apply output gate: sigmoid(gate) * attn_output
        output = output * op.sigmoid(gate)
        return self.o_proj(output)


# ============================================================================
# GatedDeltaNet TIR kernel
# ============================================================================


def create_gated_delta_net_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Creates a TIR function for the GatedDeltaNet recurrent computation.

    Thread-per-column design: each thread owns one column of the state matrix.
    State S is (key_head_dim x value_head_dim) per head, accumulated in fp32.

    Supports arbitrary sequence length via an inner `for t in range(seq_len)` loop,
    matching RWKV6's approach. During prefill (seq_len > 1), the recurrence accumulates
    state across all tokens sequentially. During decode (seq_len = 1), it's a single step.

    For GVA (num_value_heads > num_key_heads), Q/K are expanded via repeat.
    The kernel operates on value_heads (the larger dimension).
    """
    heads_per_group = num_value_heads // num_key_heads  # 1 for 0.8B, 2 for 4B
    K = key_head_dim  # 128
    V = value_head_dim  # 128

    @T.prim_func
    def gdn_func(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,  # exp(g), already exponentiated
        beta_handle: T.handle,  # sigmoid(beta_raw)
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        # q, k: (batch, seq_len, key_heads, K)
        q_buf = T.match_buffer(q_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        # v: (batch, seq_len, value_heads, V)
        v_buf = T.match_buffer(v_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype)
        # gate and beta: (batch, seq_len, value_heads)
        gate_buf = T.match_buffer(
            gate_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        # State: per value_head, K x V matrix in fp32
        state_in_buf = T.match_buffer(
            state_in_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )
        # Outputs: out in fp32 for numerical stability (cast to model dtype by caller)
        out_buf = T.match_buffer(
            out_handle, (batch_size, seq_len, num_value_heads, V), dtype="float32"
        )
        state_out_buf = T.match_buffer(
            state_out_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    kh = h_idx // heads_per_group

                    # Init state from state_in
                    for row in range(K):
                        with T.sblock("init_state"):
                            vb, vh, vr, vc = T.axis.remap("SSSS", [b_idx, h_idx, row, col])
                            state_out_buf[vb, vh, vr, vc] = state_in_buf[vb, vh, vr, vc]

                    # Sequential loop over tokens (like RWKV6)
                    for t in range(seq_len):
                        # 1. Decay state: S = gate * S
                        for row in range(K):
                            with T.sblock("decay"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vr = T.axis.opaque(K, row)
                                vc = T.axis.spatial(V, col)
                                state_out_buf[vb, vh, vr, vc] = (
                                    state_out_buf[vb, vh, vr, vc] * gate_buf[vb, vt, vh]
                                )

                        # 2. Compute dot(S[:, col], k[:]) → out_buf (fp32)
                        with T.sblock("dot_sk_init"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vc = T.axis.spatial(V, col)
                            out_buf[vb, vt, vh, vc] = T.float32(0)

                        for row in range(K):
                            with T.sblock("dot_sk"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vr = T.axis.opaque(K, row)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vc = T.axis.spatial(V, col)
                                out_buf[vb, vt, vh, vc] = out_buf[vb, vt, vh, vc] + state_out_buf[
                                    vb, vh, vr, vc
                                ] * T.cast(k_buf[vb, vt, kh, vr], "float32")

                        # 3. Delta rule: S += k * beta * (v - dot_sk)
                        for row in range(K):
                            with T.sblock("delta"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vr = T.axis.opaque(K, row)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vc = T.axis.spatial(V, col)
                                state_out_buf[vb, vh, vr, vc] = state_out_buf[
                                    vb, vh, vr, vc
                                ] + T.cast(k_buf[vb, vt, kh, vr], "float32") * beta_buf[
                                    vb, vt, vh
                                ] * (
                                    T.cast(v_buf[vb, vt, vh, vc], "float32")
                                    - out_buf[vb, vt, vh, vc]
                                )

                        # 4. Output: o[t, col] = dot(S_updated[:, col], q[t, :]) * scale
                        with T.sblock("out_init"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vc = T.axis.spatial(V, col)
                            out_buf[vb, vt, vh, vc] = T.float32(0)

                        for row in range(K):
                            with T.sblock("dot_sq"):
                                vb = T.axis.spatial(batch_size, b_idx)
                                vt = T.axis.opaque(seq_len, t)
                                vr = T.axis.opaque(K, row)
                                vh = T.axis.spatial(num_value_heads, h_idx)
                                vc = T.axis.spatial(V, col)
                                out_buf[vb, vt, vh, vc] = out_buf[vb, vt, vh, vc] + state_out_buf[
                                    vb, vh, vr, vc
                                ] * T.cast(q_buf[vb, vt, kh, vr], "float32")

                        # 5. Apply scale
                        with T.sblock("scale"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vc = T.axis.spatial(V, col)
                            out_buf[vb, vt, vh, vc] = out_buf[vb, vt, vh, vc] * T.float32(
                                1.0 / math.sqrt(K)
                            )

    return gdn_func


def create_gated_delta_net_decode_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Creates a decode-specialized GatedDeltaNet recurrent kernel.

    The generic kernel supports multi-token prefill by first materializing a
    decayed `state_out` and then doing separate state-update/output passes. For
    decode, seq_len is fixed to 1, so we can compute directly from `state_in`,
    write `state_out` once, and accumulate the output in the same row pass.
    """
    heads_per_group = num_value_heads // num_key_heads
    K = key_head_dim
    V = value_head_dim

    @T.prim_func
    def gdn_decode_func(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,
        beta_handle: T.handle,
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size = T.int64()
        q_buf = T.match_buffer(q_handle, (batch_size, 1, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, 1, num_key_heads, K), dtype=dtype)
        v_buf = T.match_buffer(v_handle, (batch_size, 1, num_value_heads, V), dtype=dtype)
        gate_buf = T.match_buffer(gate_handle, (batch_size, 1, num_value_heads), dtype="float32")
        beta_buf = T.match_buffer(beta_handle, (batch_size, 1, num_value_heads), dtype="float32")
        state_in_buf = T.match_buffer(
            state_in_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )
        out_buf = T.match_buffer(out_handle, (batch_size, 1, num_value_heads, V), dtype="float32")
        state_out_buf = T.match_buffer(
            state_out_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )

        dot_sk = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")
        dot_sq = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    kh = h_idx // heads_per_group

                    with T.sblock("dot_sk_init"):
                        dot_sk[0] = T.float32(0)

                    for row in range(K):
                        with T.sblock("dot_sk_decode"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vr = T.axis.opaque(K, row)
                            vc = T.axis.spatial(V, col)
                            dot_sk[0] = dot_sk[0] + state_in_buf[
                                vb, vh, vr, vc
                            ] * gate_buf[vb, 0, vh] * T.cast(k_buf[vb, 0, kh, vr], "float32")

                    with T.sblock("dot_sq_init"):
                        dot_sq[0] = T.float32(0)

                    for row in range(K):
                        with T.sblock("delta_and_dot_sq_decode"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vr = T.axis.opaque(K, row)
                            vc = T.axis.spatial(V, col)
                            decayed = state_in_buf[vb, vh, vr, vc] * gate_buf[vb, 0, vh]
                            new_state = decayed + T.cast(k_buf[vb, 0, kh, vr], "float32") * (
                                beta_buf[vb, 0, vh]
                            ) * (T.cast(v_buf[vb, 0, vh, vc], "float32") - dot_sk[0])
                            state_out_buf[vb, vh, vr, vc] = new_state
                            dot_sq[0] = dot_sq[0] + new_state * T.cast(
                                q_buf[vb, 0, kh, vr], "float32"
                            )

                    with T.sblock("scale_decode"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        vc = T.axis.spatial(V, col)
                        out_buf[vb, 0, vh, vc] = dot_sq[0] * T.float32(1.0 / math.sqrt(K))

    return gdn_decode_func


def create_gated_delta_net_decode_state_storage_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Creates a decode-specialized GDN kernel that updates RNN storage in place."""
    heads_per_group = num_value_heads // num_key_heads
    K = key_head_dim
    V = value_head_dim

    @T.prim_func
    def gdn_decode_state_storage_func(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,
        beta_handle: T.handle,
        state_storage_handle: T.handle,
        seq_slot_ids_handle: T.handle,
        history_slot_ids_handle: T.handle,
        out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size = T.int64()
        max_batch_size = T.int64()
        max_history = T.int64()
        q_buf = T.match_buffer(q_handle, (batch_size, 1, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, 1, num_key_heads, K), dtype=dtype)
        v_buf = T.match_buffer(v_handle, (batch_size, 1, num_value_heads, V), dtype=dtype)
        gate_buf = T.match_buffer(gate_handle, (batch_size, 1, num_value_heads), dtype="float32")
        beta_buf = T.match_buffer(beta_handle, (batch_size, 1, num_value_heads), dtype="float32")
        state_storage_buf = T.match_buffer(
            state_storage_handle,
            (max_batch_size, max_history, num_value_heads, K, V),
            dtype="float32",
        )
        seq_slot_ids_buf = T.match_buffer(seq_slot_ids_handle, (batch_size,), dtype="int32")
        history_slot_ids_buf = T.match_buffer(history_slot_ids_handle, (batch_size,), dtype="int32")
        out_buf = T.match_buffer(out_handle, (batch_size, 1, num_value_heads, V), dtype="float32")

        dot_sk = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")
        dot_sq = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            seq_slot = T.cast(seq_slot_ids_buf[b_idx], "int64")
            history_slot = T.cast(history_slot_ids_buf[b_idx], "int64")
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    kh = h_idx // heads_per_group

                    with T.sblock("dot_sk_storage_init"):
                        dot_sk[0] = T.float32(0)

                    for row in range(K):
                        with T.sblock("dot_sk_storage_decode"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vr = T.axis.opaque(K, row)
                            vc = T.axis.spatial(V, col)
                            dot_sk[0] = dot_sk[0] + state_storage_buf[
                                seq_slot, history_slot, vh, vr, vc
                            ] * gate_buf[vb, 0, vh] * T.cast(k_buf[vb, 0, kh, vr], "float32")

                    with T.sblock("dot_sq_storage_init"):
                        dot_sq[0] = T.float32(0)

                    for row in range(K):
                        with T.sblock("delta_and_dot_sq_storage_decode"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vr = T.axis.opaque(K, row)
                            vc = T.axis.spatial(V, col)
                            decayed = (
                                state_storage_buf[seq_slot, history_slot, vh, vr, vc]
                                * gate_buf[vb, 0, vh]
                            )
                            new_state = decayed + T.cast(k_buf[vb, 0, kh, vr], "float32") * (
                                beta_buf[vb, 0, vh]
                            ) * (T.cast(v_buf[vb, 0, vh, vc], "float32") - dot_sk[0])
                            state_storage_buf[seq_slot, history_slot, vh, vr, vc] = new_state
                            dot_sq[0] = dot_sq[0] + new_state * T.cast(
                                q_buf[vb, 0, kh, vr], "float32"
                            )

                    with T.sblock("scale_storage_decode"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        vc = T.axis.spatial(V, col)
                        out_buf[vb, 0, vh, vc] = dot_sq[0] * T.float32(1.0 / math.sqrt(K))

    return gdn_decode_state_storage_func


def create_gated_delta_net_packed_decode_state_storage_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Creates a packed decode GDN kernel that folds prep into recurrence.

    The decode path is always a single token. This kernel consumes post-conv
    packed QKV plus alpha/beta inputs, computes SiLU, Q/K L2 normalization,
    gate/beta, recurrent update, output, and z SiLU in one launch.
    """
    heads_per_group = num_value_heads // num_key_heads
    K = key_head_dim
    V = value_head_dim
    q_dim = num_key_heads * K
    k_dim = num_key_heads * K
    v_dim = num_value_heads * V
    qkv_dim = q_dim + k_dim + v_dim
    z_dim = v_dim

    @T.prim_func
    def gdn_packed_decode_state_storage_func(
        qkv_handle: T.handle,
        z_handle: T.handle,
        alpha_handle: T.handle,
        beta_raw_handle: T.handle,
        a_log_handle: T.handle,
        dt_bias_handle: T.handle,
        state_storage_handle: T.handle,
        seq_slot_ids_handle: T.handle,
        history_slot_ids_handle: T.handle,
        out_handle: T.handle,
        z_silu_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size = T.int64()
        max_batch_size = T.int64()
        max_history = T.int64()
        qkv_buf = T.match_buffer(qkv_handle, (batch_size, 1, qkv_dim), dtype=dtype)
        z_buf = T.match_buffer(z_handle, (batch_size, 1, z_dim), dtype=dtype)
        alpha_buf = T.match_buffer(alpha_handle, (batch_size, 1, num_value_heads), dtype=dtype)
        beta_raw_buf = T.match_buffer(
            beta_raw_handle, (batch_size, 1, num_value_heads), dtype=dtype
        )
        a_log_buf = T.match_buffer(a_log_handle, (num_value_heads,), dtype="float32")
        dt_bias_buf = T.match_buffer(dt_bias_handle, (num_value_heads,), dtype="float32")
        state_storage_buf = T.match_buffer(
            state_storage_handle,
            (max_batch_size, max_history, num_value_heads, K, V),
            dtype="float32",
        )
        seq_slot_ids_buf = T.match_buffer(seq_slot_ids_handle, (batch_size,), dtype="int32")
        history_slot_ids_buf = T.match_buffer(history_slot_ids_handle, (batch_size,), dtype="int32")
        out_buf = T.match_buffer(out_handle, (batch_size, 1, num_value_heads, V), dtype="float32")
        z_silu_out_buf = T.match_buffer(z_silu_out_handle, (batch_size, 1, z_dim), dtype=dtype)

        q_sum_shared = T.sblock_alloc_buffer(
            (batch_size, num_value_heads), dtype="float32", scope="shared"
        )
        k_sum_shared = T.sblock_alloc_buffer(
            (batch_size, num_value_heads), dtype="float32", scope="shared"
        )
        q_norm_shared = T.sblock_alloc_buffer(
            (batch_size, num_value_heads, K), dtype="float32", scope="shared"
        )
        k_norm_shared = T.sblock_alloc_buffer(
            (batch_size, num_value_heads, K), dtype="float32", scope="shared"
        )
        gate_shared = T.sblock_alloc_buffer(
            (batch_size, num_value_heads), dtype="float32", scope="shared"
        )
        beta_shared = T.sblock_alloc_buffer(
            (batch_size, num_value_heads), dtype="float32", scope="shared"
        )
        q_sum_local = T.sblock_alloc_buffer(
            (K, batch_size, num_value_heads), dtype="float32", scope="local"
        )
        k_sum_local = T.sblock_alloc_buffer(
            (K, batch_size, num_value_heads), dtype="float32", scope="local"
        )
        dot_sk = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")
        dot_sq = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")
        v_silu_local = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            seq_slot = T.cast(seq_slot_ids_buf[b_idx], "int64")
            history_slot = T.cast(history_slot_ids_buf[b_idx], "int64")
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                kh = h_idx // heads_per_group
                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("qk_sum_packed"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        vr = T.axis.spatial(K, tx)
                        q_raw = T.cast(qkv_buf[vb, 0, kh * K + vr], "float32")
                        k_raw = T.cast(qkv_buf[vb, 0, q_dim + kh * K + vr], "float32")
                        q_silu = q_raw / (T.float32(1) + T.exp(-q_raw))
                        k_silu = k_raw / (T.float32(1) + T.exp(-k_raw))
                        q_sum_local[vr, vb, vh] = q_silu * q_silu
                        k_sum_local[vr, vb, vh] = k_silu * k_silu

                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("q_sum_reduce_packed"):
                        vr = T.axis.reduce(K, tx)
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        T.reads(q_sum_local[vr, vb, vh])
                        T.writes(q_sum_shared[vb, vh])
                        with T.init():
                            q_sum_shared[vb, vh] = T.float32(0)
                        q_sum_shared[vb, vh] = q_sum_shared[vb, vh] + q_sum_local[vr, vb, vh]

                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("k_sum_reduce_packed"):
                        vr = T.axis.reduce(K, tx)
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        T.reads(k_sum_local[vr, vb, vh])
                        T.writes(k_sum_shared[vb, vh])
                        with T.init():
                            k_sum_shared[vb, vh] = T.float32(0)
                        k_sum_shared[vb, vh] = k_sum_shared[vb, vh] + k_sum_local[vr, vb, vh]

                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("qk_norm_packed"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        vr = T.axis.spatial(K, tx)
                        q_raw = T.cast(qkv_buf[vb, 0, kh * K + vr], "float32")
                        k_raw = T.cast(qkv_buf[vb, 0, q_dim + kh * K + vr], "float32")
                        q_silu = q_raw / (T.float32(1) + T.exp(-q_raw))
                        k_silu = k_raw / (T.float32(1) + T.exp(-k_raw))
                        q_norm_shared[vb, vh, vr] = q_silu * T.rsqrt(
                            q_sum_shared[vb, vh] + T.float32(1e-6)
                        )
                        k_norm_shared[vb, vh, vr] = k_silu * T.rsqrt(
                            k_sum_shared[vb, vh] + T.float32(1e-6)
                        )

                for tx in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("gate_beta_packed_decode"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        alpha = T.cast(alpha_buf[vb, 0, vh], "float32") + dt_bias_buf[vh]
                        softplus = T.if_then_else(
                            alpha > T.float32(20),
                            alpha,
                            T.log(T.float32(1) + T.exp(alpha)),
                        )
                        gate_shared[vb, vh] = T.exp(-T.exp(a_log_buf[vh]) * softplus)
                        beta_raw = T.cast(beta_raw_buf[vb, 0, vh], "float32")
                        beta_shared[vb, vh] = T.float32(1) / (T.float32(1) + T.exp(-beta_raw))

                for col in T.thread_binding(V, thread="threadIdx.x"):
                    with T.sblock("v_silu_packed_decode"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        vc = T.axis.spatial(V, col)
                        v_raw_col = T.cast(qkv_buf[vb, 0, q_dim + k_dim + vh * V + vc], "float32")
                        v_silu_local[0] = v_raw_col / (T.float32(1) + T.exp(-v_raw_col))

                    with T.sblock("dot_sk_packed_init"):
                        dot_sk[0] = T.float32(0)

                    for row in range(K):
                        with T.sblock("dot_sk_packed_decode"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vr = T.axis.opaque(K, row)
                            vc = T.axis.spatial(V, col)
                            dot_sk[0] = dot_sk[0] + state_storage_buf[
                                seq_slot, history_slot, vh, vr, vc
                            ] * gate_shared[vb, vh] * k_norm_shared[vb, vh, vr]

                    with T.sblock("dot_sq_packed_init"):
                        dot_sq[0] = T.float32(0)

                    for row in range(K):
                        with T.sblock("delta_and_dot_sq_packed_decode"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vh = T.axis.spatial(num_value_heads, h_idx)
                            vr = T.axis.opaque(K, row)
                            vc = T.axis.spatial(V, col)
                            decayed = (
                                state_storage_buf[seq_slot, history_slot, vh, vr, vc]
                                * gate_shared[vb, vh]
                            )
                            new_state = decayed + k_norm_shared[vb, vh, vr] * beta_shared[
                                vb, vh
                            ] * (v_silu_local[0] - dot_sk[0])
                            state_storage_buf[seq_slot, history_slot, vh, vr, vc] = new_state
                            dot_sq[0] = dot_sq[0] + new_state * q_norm_shared[vb, vh, vr]

                    with T.sblock("scale_and_z_packed_decode"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        vc = T.axis.spatial(V, col)
                        z_index = vh * V + vc
                        z_raw = T.cast(z_buf[vb, 0, z_index], "float32")
                        out_buf[vb, 0, vh, vc] = dot_sq[0] * T.float32(1.0 / math.sqrt(K))
                        z_silu_out_buf[vb, 0, z_index] = T.cast(
                            z_raw / (T.float32(1) + T.exp(-z_raw)), dtype
                        )

    return gdn_packed_decode_state_storage_func


def create_causal_conv1d_decode_state_storage_func(
    kernel_size: int,
    qkv_dim: int,
    dtype: str,
):
    """Creates a decode-specialized causal conv kernel that updates state in place."""
    ks_minus_1 = kernel_size - 1

    @T.prim_func
    def causal_conv1d_decode_state_storage_func(
        qkv_handle: T.handle,
        state_storage_handle: T.handle,
        seq_slot_ids_handle: T.handle,
        history_slot_ids_handle: T.handle,
        weight_handle: T.handle,
        out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size = T.int64()
        max_batch_size = T.int64()
        max_history = T.int64()
        qkv_buf = T.match_buffer(qkv_handle, (batch_size, 1, qkv_dim), dtype=dtype)
        state_storage_buf = T.match_buffer(
            state_storage_handle,
            (max_batch_size, max_history, ks_minus_1, qkv_dim),
            dtype=dtype,
        )
        seq_slot_ids_buf = T.match_buffer(seq_slot_ids_handle, (batch_size,), dtype="int32")
        history_slot_ids_buf = T.match_buffer(history_slot_ids_handle, (batch_size,), dtype="int32")
        weight_buf = T.match_buffer(weight_handle, (qkv_dim, 1, kernel_size), dtype=dtype)
        out_buf = T.match_buffer(out_handle, (batch_size, 1, qkv_dim), dtype=dtype)

        acc = T.sblock_alloc_buffer((1,), dtype=dtype, scope="local")

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            seq_slot = T.cast(seq_slot_ids_buf[b_idx], "int64")
            history_slot = T.cast(history_slot_ids_buf[b_idx], "int64")
            next_history_slot = T.cast(
                (history_slot_ids_buf[b_idx] + T.int32(1)) % T.cast(max_history, "int32"),
                "int64",
            )
            for d_idx in T.thread_binding(qkv_dim, thread="blockIdx.x"):
                with T.sblock("conv_storage_init"):
                    acc[0] = T.cast(0, dtype)

                for kk in range(kernel_size):
                    with T.sblock("conv_storage_decode"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vd = T.axis.spatial(qkv_dim, d_idx)
                        vk = T.axis.opaque(kernel_size, kk)
                        x = T.if_then_else(
                            vk < ks_minus_1,
                            state_storage_buf[seq_slot, history_slot, vk, vd],
                            qkv_buf[vb, 0, vd],
                        )
                        acc[0] = acc[0] + x * weight_buf[vd, 0, vk]

                with T.sblock("conv_storage_output"):
                    vb = T.axis.spatial(batch_size, b_idx)
                    vd = T.axis.spatial(qkv_dim, d_idx)
                    out_buf[vb, 0, vd] = acc[0]

                for pos in range(ks_minus_1):
                    with T.sblock("conv_storage_update"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vd = T.axis.spatial(qkv_dim, d_idx)
                        vp = T.axis.spatial(ks_minus_1, pos)
                        new_value = T.if_then_else(
                            vp + 1 < ks_minus_1,
                            state_storage_buf[seq_slot, history_slot, vp + 1, vd],
                            qkv_buf[vb, 0, vd],
                        )
                        state_storage_buf[seq_slot, next_history_slot, vp, vd] = new_value

    return causal_conv1d_decode_state_storage_func


def create_gdn_decode_prepare_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Fuse decode-only GDN preparation into one kernel.

    This covers the small decode kernels between causal conv and recurrent GDN:
    SiLU(QKV), Q/K split and L2 normalization, V split, gate, beta, and SiLU(Z).
    """
    K = key_head_dim
    V = value_head_dim
    q_dim = num_key_heads * K
    k_dim = num_key_heads * K
    v_dim = num_value_heads * V
    qkv_dim = q_dim + k_dim + v_dim
    z_dim = v_dim

    @T.prim_func
    def gdn_decode_prepare_func(
        qkv_handle: T.handle,
        z_handle: T.handle,
        alpha_handle: T.handle,
        beta_raw_handle: T.handle,
        a_log_handle: T.handle,
        dt_bias_handle: T.handle,
        q_out_handle: T.handle,
        k_out_handle: T.handle,
        v_out_handle: T.handle,
        gate_out_handle: T.handle,
        beta_out_handle: T.handle,
        z_silu_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size = T.int64()
        qkv_buf = T.match_buffer(qkv_handle, (batch_size, 1, qkv_dim), dtype=dtype)
        z_buf = T.match_buffer(z_handle, (batch_size, 1, z_dim), dtype=dtype)
        alpha_buf = T.match_buffer(alpha_handle, (batch_size, 1, num_value_heads), dtype=dtype)
        beta_raw_buf = T.match_buffer(
            beta_raw_handle, (batch_size, 1, num_value_heads), dtype=dtype
        )
        a_log_buf = T.match_buffer(a_log_handle, (num_value_heads,), dtype="float32")
        dt_bias_buf = T.match_buffer(dt_bias_handle, (num_value_heads,), dtype="float32")
        q_out_buf = T.match_buffer(q_out_handle, (batch_size, 1, num_key_heads, K), dtype=dtype)
        k_out_buf = T.match_buffer(k_out_handle, (batch_size, 1, num_key_heads, K), dtype=dtype)
        v_out_buf = T.match_buffer(
            v_out_handle, (batch_size, 1, num_value_heads, V), dtype=dtype
        )
        gate_out_buf = T.match_buffer(
            gate_out_handle, (batch_size, 1, num_value_heads), dtype="float32"
        )
        beta_out_buf = T.match_buffer(
            beta_out_handle, (batch_size, 1, num_value_heads), dtype="float32"
        )
        z_silu_out_buf = T.match_buffer(z_silu_out_handle, (batch_size, 1, z_dim), dtype=dtype)

        q_sum_shared = T.sblock_alloc_buffer(
            (batch_size, num_key_heads), dtype="float32", scope="shared"
        )
        k_sum_shared = T.sblock_alloc_buffer(
            (batch_size, num_key_heads), dtype="float32", scope="shared"
        )
        q_sum_local = T.sblock_alloc_buffer(
            (K, batch_size, num_key_heads), dtype="float32", scope="local"
        )
        k_sum_local = T.sblock_alloc_buffer(
            (K, batch_size, num_key_heads), dtype="float32", scope="local"
        )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_key_heads, thread="blockIdx.x"):
                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("qk_sum"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_key_heads, h_idx)
                        vr = T.axis.spatial(K, tx)
                        q_raw = T.cast(qkv_buf[vb, 0, vh * K + vr], "float32")
                        k_raw = T.cast(qkv_buf[vb, 0, q_dim + vh * K + vr], "float32")
                        q_silu = q_raw / (T.float32(1) + T.exp(-q_raw))
                        k_silu = k_raw / (T.float32(1) + T.exp(-k_raw))
                        q_sum_local[vr, vb, vh] = q_silu * q_silu
                        k_sum_local[vr, vb, vh] = k_silu * k_silu

                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("q_sum_reduce"):
                        vr = T.axis.reduce(K, tx)
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_key_heads, h_idx)
                        T.reads(q_sum_local[vr, vb, vh])
                        T.writes(q_sum_shared[vb, vh])
                        with T.init():
                            q_sum_shared[vb, vh] = T.float32(0)
                        q_sum_shared[vb, vh] = q_sum_shared[vb, vh] + q_sum_local[vr, vb, vh]

                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("k_sum_reduce"):
                        vr = T.axis.reduce(K, tx)
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_key_heads, h_idx)
                        T.reads(k_sum_local[vr, vb, vh])
                        T.writes(k_sum_shared[vb, vh])
                        with T.init():
                            k_sum_shared[vb, vh] = T.float32(0)
                        k_sum_shared[vb, vh] = k_sum_shared[vb, vh] + k_sum_local[vr, vb, vh]

                for tx in T.thread_binding(K, thread="threadIdx.x"):
                    with T.sblock("qk_write"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vh = T.axis.spatial(num_key_heads, h_idx)
                        vr = T.axis.spatial(K, tx)
                        q_raw = T.cast(qkv_buf[vb, 0, vh * K + vr], "float32")
                        k_raw = T.cast(qkv_buf[vb, 0, q_dim + vh * K + vr], "float32")
                        q_silu = q_raw / (T.float32(1) + T.exp(-q_raw))
                        k_silu = k_raw / (T.float32(1) + T.exp(-k_raw))
                        q_out_buf[vb, 0, vh, vr] = T.cast(
                            q_silu * T.rsqrt(q_sum_shared[vb, vh] + T.float32(1e-6)), dtype
                        )
                        k_out_buf[vb, 0, vh, vr] = T.cast(
                            k_silu * T.rsqrt(k_sum_shared[vb, vh] + T.float32(1e-6)), dtype
                        )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for d_idx in T.thread_binding(v_dim, thread="blockIdx.x"):
                with T.sblock("v_z_write"):
                    vb = T.axis.spatial(batch_size, b_idx)
                    vd = T.axis.spatial(v_dim, d_idx)
                    raw_v = T.cast(qkv_buf[vb, 0, q_dim + k_dim + vd], "float32")
                    raw_z = T.cast(z_buf[vb, 0, vd], "float32")
                    vh = vd // V
                    vc = vd % V
                    v_out_buf[vb, 0, vh, vc] = T.cast(
                        raw_v / (T.float32(1) + T.exp(-raw_v)), dtype
                    )
                    z_silu_out_buf[vb, 0, vd] = T.cast(
                        raw_z / (T.float32(1) + T.exp(-raw_z)), dtype
                    )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                with T.sblock("gate_beta_write"):
                    vb = T.axis.spatial(batch_size, b_idx)
                    vh = T.axis.spatial(num_value_heads, h_idx)
                    alpha = T.cast(alpha_buf[vb, 0, vh], "float32") + dt_bias_buf[vh]
                    softplus = T.if_then_else(
                        alpha > T.float32(20),
                        alpha,
                        T.log(T.float32(1) + T.exp(alpha)),
                    )
                    beta_raw = T.cast(beta_raw_buf[vb, 0, vh], "float32")
                    gate_out_buf[vb, 0, vh] = T.exp(-T.exp(a_log_buf[vh]) * softplus)
                    beta_out_buf[vb, 0, vh] = T.float32(1) / (
                        T.float32(1) + T.exp(-beta_raw)
                    )

    return gdn_decode_prepare_func


def create_gdn_prefill_prepare_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Fuse prefill GDN preparation into one kernel.

    This mirrors the decode-only preparation kernel, but supports arbitrary
    prefill length: SiLU(QKV), Q/K split and L2 normalization, V split, gate,
    beta, and SiLU(Z).
    """
    K = key_head_dim
    V = value_head_dim
    q_dim = num_key_heads * K
    k_dim = num_key_heads * K
    v_dim = num_value_heads * V
    qkv_dim = q_dim + k_dim + v_dim
    z_dim = v_dim

    @T.prim_func
    def gdn_prefill_prepare_func(
        qkv_handle: T.handle,
        z_handle: T.handle,
        alpha_handle: T.handle,
        beta_raw_handle: T.handle,
        a_log_handle: T.handle,
        dt_bias_handle: T.handle,
        q_out_handle: T.handle,
        k_out_handle: T.handle,
        v_out_handle: T.handle,
        gate_out_handle: T.handle,
        beta_out_handle: T.handle,
        z_silu_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        qkv_buf = T.match_buffer(qkv_handle, (batch_size, seq_len, qkv_dim), dtype=dtype)
        z_buf = T.match_buffer(z_handle, (batch_size, seq_len, z_dim), dtype=dtype)
        alpha_buf = T.match_buffer(
            alpha_handle, (batch_size, seq_len, num_value_heads), dtype=dtype
        )
        beta_raw_buf = T.match_buffer(
            beta_raw_handle, (batch_size, seq_len, num_value_heads), dtype=dtype
        )
        a_log_buf = T.match_buffer(a_log_handle, (num_value_heads,), dtype="float32")
        dt_bias_buf = T.match_buffer(dt_bias_handle, (num_value_heads,), dtype="float32")
        q_out_buf = T.match_buffer(
            q_out_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype
        )
        k_out_buf = T.match_buffer(
            k_out_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype
        )
        v_out_buf = T.match_buffer(
            v_out_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype
        )
        gate_out_buf = T.match_buffer(
            gate_out_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_out_buf = T.match_buffer(
            beta_out_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        z_silu_out_buf = T.match_buffer(
            z_silu_out_handle, (batch_size, seq_len, z_dim), dtype=dtype
        )

        q_sum_shared = T.sblock_alloc_buffer((1,), dtype="float32", scope="shared")
        k_sum_shared = T.sblock_alloc_buffer((1,), dtype="float32", scope="shared")
        q_sum_local = T.sblock_alloc_buffer((K,), dtype="float32", scope="local")
        k_sum_local = T.sblock_alloc_buffer((K,), dtype="float32", scope="local")

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.z"):
            for t_idx in T.thread_binding(seq_len, thread="blockIdx.y"):
                for h_idx in T.thread_binding(num_key_heads, thread="blockIdx.x"):
                    for tx in T.thread_binding(K, thread="threadIdx.x"):
                        with T.sblock("qk_sum"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.spatial(seq_len, t_idx)
                            vh = T.axis.spatial(num_key_heads, h_idx)
                            vr = T.axis.spatial(K, tx)
                            q_raw = T.cast(qkv_buf[vb, vt, vh * K + vr], "float32")
                            k_raw = T.cast(qkv_buf[vb, vt, q_dim + vh * K + vr], "float32")
                            q_silu = q_raw / (T.float32(1) + T.exp(-q_raw))
                            k_silu = k_raw / (T.float32(1) + T.exp(-k_raw))
                            q_sum_local[vr] = q_silu * q_silu
                            k_sum_local[vr] = k_silu * k_silu

                    for tx in T.thread_binding(K, thread="threadIdx.x"):
                        with T.sblock("q_sum_reduce"):
                            vr = T.axis.reduce(K, tx)
                            T.reads(q_sum_local[vr])
                            T.writes(q_sum_shared[0])
                            with T.init():
                                q_sum_shared[0] = T.float32(0)
                            q_sum_shared[0] = q_sum_shared[0] + q_sum_local[vr]

                    for tx in T.thread_binding(K, thread="threadIdx.x"):
                        with T.sblock("k_sum_reduce"):
                            vr = T.axis.reduce(K, tx)
                            T.reads(k_sum_local[vr])
                            T.writes(k_sum_shared[0])
                            with T.init():
                                k_sum_shared[0] = T.float32(0)
                            k_sum_shared[0] = k_sum_shared[0] + k_sum_local[vr]

                    for tx in T.thread_binding(K, thread="threadIdx.x"):
                        with T.sblock("qk_write"):
                            vb = T.axis.spatial(batch_size, b_idx)
                            vt = T.axis.spatial(seq_len, t_idx)
                            vh = T.axis.spatial(num_key_heads, h_idx)
                            vr = T.axis.spatial(K, tx)
                            q_raw = T.cast(qkv_buf[vb, vt, vh * K + vr], "float32")
                            k_raw = T.cast(qkv_buf[vb, vt, q_dim + vh * K + vr], "float32")
                            q_silu = q_raw / (T.float32(1) + T.exp(-q_raw))
                            k_silu = k_raw / (T.float32(1) + T.exp(-k_raw))
                            q_out_buf[vb, vt, vh, vr] = T.cast(
                                q_silu * T.rsqrt(q_sum_shared[0] + T.float32(1e-6)),
                                dtype,
                            )
                            k_out_buf[vb, vt, vh, vr] = T.cast(
                                k_silu * T.rsqrt(k_sum_shared[0] + T.float32(1e-6)),
                                dtype,
                            )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.z"):
            for t_idx in T.thread_binding(seq_len, thread="blockIdx.y"):
                for d_idx in T.thread_binding(v_dim, thread="blockIdx.x"):
                    with T.sblock("v_z_write"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vt = T.axis.spatial(seq_len, t_idx)
                        vd = T.axis.spatial(v_dim, d_idx)
                        raw_v = T.cast(qkv_buf[vb, vt, q_dim + k_dim + vd], "float32")
                        raw_z = T.cast(z_buf[vb, vt, vd], "float32")
                        vh = vd // V
                        vc = vd % V
                        v_out_buf[vb, vt, vh, vc] = T.cast(
                            raw_v / (T.float32(1) + T.exp(-raw_v)), dtype
                        )
                        z_silu_out_buf[vb, vt, vd] = T.cast(
                            raw_z / (T.float32(1) + T.exp(-raw_z)), dtype
                        )

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.z"):
            for t_idx in T.thread_binding(seq_len, thread="blockIdx.y"):
                for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                    with T.sblock("gate_beta_write"):
                        vb = T.axis.spatial(batch_size, b_idx)
                        vt = T.axis.spatial(seq_len, t_idx)
                        vh = T.axis.spatial(num_value_heads, h_idx)
                        alpha = T.cast(alpha_buf[vb, vt, vh], "float32") + dt_bias_buf[vh]
                        softplus = T.if_then_else(
                            alpha > T.float32(20),
                            alpha,
                            T.log(T.float32(1) + T.exp(alpha)),
                        )
                        beta_raw = T.cast(beta_raw_buf[vb, vt, vh], "float32")
                        gate_out_buf[vb, vt, vh] = T.exp(-T.exp(a_log_buf[vh]) * softplus)
                        beta_out_buf[vb, vt, vh] = T.float32(1) / (
                            T.float32(1) + T.exp(-beta_raw)
                        )

    return gdn_prefill_prepare_func


def _direct_conv_state_dtype(default_dtype: str) -> str:
    dtype = os.environ.get("MLC_QWEN35_DIRECT_CONV_STATE_DTYPE")
    if dtype:
        return dtype
    # Qwen3.5 configs may keep the model dtype as bf16 while q0f16 compilation
    # lowers linear activations to fp16. The direct conv state must match qkv.
    return "float16" if default_dtype == "bfloat16" else default_dtype


def _is_decode_seq_len_one(seq_len: Any) -> bool:
    return isinstance(seq_len, int) and seq_len == 1


def _allow_direct_rnn_state_for_decode(config: Qwen35Config, batch_size: Any, seq_len: Any) -> bool:
    if os.environ.get("MLC_QWEN35_DIRECT_RNN_STATE", "0") != "1":
        return False
    if not _is_decode_seq_len_one(seq_len):
        return False
    if isinstance(batch_size, int):
        return batch_size == 1
    return getattr(config, "max_batch_size", 1) == 1


# ============================================================================
# GatedDeltaNet Linear Attention Layer
# ============================================================================


class Qwen35GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention layer."""

    def __init__(self, config: Qwen35Config, linear_layer_idx: int):
        self.config = config
        self.linear_layer_idx = linear_layer_idx  # index among linear layers only
        self.key_head_dim = config.linear_key_head_dim  # 128
        self.value_head_dim = config.linear_value_head_dim  # 128
        self.num_key_heads = config.linear_num_key_heads  # 16
        self.num_value_heads = config.linear_num_value_heads  # 16 or 32
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype

        qkv_dim = (
            (self.num_key_heads * self.key_head_dim)
            + (self.num_key_heads * self.key_head_dim)
            + (self.num_value_heads * self.value_head_dim)
        )
        z_dim = self.num_value_heads * self.value_head_dim
        gate_dim = self.num_value_heads

        # HF stores these as four separate matrices. MLC packs them into one
        # projection so decode pays for one larger GEMM instead of four launches.
        self.in_proj = nn.Linear(
            config.hidden_size,
            qkv_dim + z_dim + gate_dim + gate_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.num_value_heads * self.value_head_dim, config.hidden_size, bias=False
        )

        # Causal depthwise Conv1D kernel
        self.conv1d_weight = nn.Parameter(
            (qkv_dim, 1, config.linear_conv_kernel_dim),
        )

        # Decay parameters (no .weight suffix in HF)
        self.A_log = nn.Parameter((self.num_value_heads,))
        self.dt_bias = nn.Parameter((self.num_value_heads,))

        # Output gating norm — per-head RMSNorm (shared weight across heads)
        self.norm = nn.RMSNorm(self.value_head_dim, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        state: RNNState,
        seq_slot_ids: Optional[Tensor] = None,
        history_slot_ids: Optional[Tensor] = None,
        state_storage: Optional[Tensor] = None,
        conv_state_storage: Optional[Tensor] = None,
    ) -> Tuple[Tensor, RNNState]:  # noqa: UP006
        """Forward using RNNState (for MLCEngine batch methods)."""
        b, s, _ = hidden_states.shape
        K = self.key_head_dim
        V = self.value_head_dim
        n_kh = self.num_key_heads
        n_vh = self.num_value_heads
        layer_idx = self.linear_layer_idx
        qkv_dim = n_kh * K + n_kh * K + n_vh * V

        use_direct_state_storage = _allow_direct_rnn_state_for_decode(self.config, b, s)
        use_direct_conv_storage = (
            use_direct_state_storage
            and os.environ.get("MLC_QWEN35_DIRECT_CONV_STATE", "0") == "1"
        )
        direct_conv_state_dtype = _direct_conv_state_dtype(self.dtype)
        direct_state_storage_batch = int(
            os.environ.get(
                "MLC_QWEN35_DIRECT_RNN_STORAGE_BATCH",
                str(self.config.max_batch_size),
            )
        )
        direct_state_max_history = int(os.environ.get("MLC_QWEN35_DIRECT_RNN_MAX_HISTORY", "1"))
        # Read conv state before the layer compute and write it back after the
        # output is computed. The matrix recurrent state can optionally use a
        # direct-storage decode path below.
        if use_direct_state_storage:
            if seq_slot_ids is None:
                seq_slot_ids = state.get_seq_slot_ids(b)
            if history_slot_ids is None:
                history_slot_ids = state.get_history_slot_ids(b)
            state_in_layer = None
        else:
            state_in_layer = state.get(layer_idx, 0, (b, n_vh, K, V), "float32")

        if use_direct_conv_storage:
            if conv_state_storage is None:
                conv_state_storage = state.get_storage(
                    layer_idx,
                    1,
                    (self.config.linear_conv_kernel_dim - 1, qkv_dim),
                    direct_conv_state_dtype,
                    direct_state_storage_batch,
                    1,
                )
            conv_state = None
        else:
            conv_state = state.get(
                layer_idx,
                1,
                (b, self.config.linear_conv_kernel_dim - 1, qkv_dim),
                self.dtype,
            )

        # Input projections. The packed weight layout is:
        # [qkv, z, alpha, beta_raw] along the output dimension.
        proj = self.in_proj(hidden_states)
        z_dim = n_vh * V
        proj_parts = op.split(proj, [qkv_dim, qkv_dim + z_dim, qkv_dim + z_dim + n_vh], axis=-1)
        qkv, z, alpha, beta_raw = proj_parts[0], proj_parts[1], proj_parts[2], proj_parts[3]

        # Causal Conv1D using existing helper logic.
        if use_direct_conv_storage:
            qkv, _ = op.tensor_ir_inplace_op(
                create_causal_conv1d_decode_state_storage_func(
                    kernel_size=self.config.linear_conv_kernel_dim,
                    qkv_dim=qkv_dim,
                    dtype=direct_conv_state_dtype,
                ),
                "causal_conv1d_state_storage",
                [qkv, conv_state_storage, seq_slot_ids, history_slot_ids, self.conv1d_weight],
                inplace_indices=[-1, 1],
                out=[
                    Tensor.placeholder([b, s, qkv_dim], self.dtype),
                    conv_state_storage,
                ],
            )
            new_conv_state = None
        else:
            qkv, new_conv_state = self._causal_conv1d_with_state(qkv, conv_state)

        q_dim = n_kh * K
        k_dim = n_kh * K
        use_packed_decode_recurrent = (
            use_direct_state_storage
            and os.environ.get("MLC_QWEN35_PACKED_GDN_DECODE", "0") == "1"
        )
        use_fused_decode_prepare = (
            use_direct_state_storage
            and not use_packed_decode_recurrent
            and os.environ.get("MLC_QWEN35_FUSED_GDN_DECODE_PREPARE", "0") == "1"
        )
        use_fused_prefill_prepare = (
            not use_direct_state_storage
            and not (isinstance(s, int) and s == 1)
            and os.environ.get("MLC_QWEN35_FUSED_GDN_PREFILL_PREPARE", "0") == "1"
        )
        chunked_gdn_chunk_size = int(os.environ.get("MLC_QWEN35_CHUNKED_GDN_CHUNK_SIZE", "64"))
        use_chunked_prefill_recurrent = (
            not use_direct_state_storage
            and not (isinstance(s, int) and s == 1)
            and K <= 128
            and chunked_gdn_chunk_size == 64
            and os.environ.get("MLC_QWEN35_CHUNKED_GDN_PREFILL", "0") == "1"
        )
        if use_packed_decode_recurrent:
            if state_storage is None:
                state_storage = state.get_storage(
                    layer_idx,
                    0,
                    (n_vh, K, V),
                    "float32",
                    direct_state_storage_batch,
                    direct_state_max_history,
                )
            out_recurrent, z_silu, _ = op.tensor_ir_inplace_op(
                create_gated_delta_net_packed_decode_state_storage_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gated_delta_net_packed_state_storage",
                [
                    qkv,
                    z,
                    alpha,
                    beta_raw,
                    self.A_log,
                    self.dt_bias,
                    state_storage,
                    seq_slot_ids,
                    history_slot_ids,
                ],
                inplace_indices=[-1, -1, 6],
                out=[
                    Tensor.placeholder([b, s, n_vh, V], "float32"),
                    Tensor.placeholder([b, s, n_vh * V], self.dtype),
                    state_storage,
                ],
            )
            state_out_layer = None
        elif use_fused_decode_prepare:
            q, k, v, gate, beta, z_silu = op.tensor_ir_op(
                create_gdn_decode_prepare_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gdn_decode_prepare",
                [qkv, z, alpha, beta_raw, self.A_log, self.dt_bias],
                [
                    Tensor.placeholder([b, s, n_kh, K], self.dtype),
                    Tensor.placeholder([b, s, n_kh, K], self.dtype),
                    Tensor.placeholder([b, s, n_vh, V], self.dtype),
                    Tensor.placeholder([b, s, n_vh], "float32"),
                    Tensor.placeholder([b, s, n_vh], "float32"),
                    Tensor.placeholder([b, s, n_vh * V], self.dtype),
                ],
            )
        elif use_fused_prefill_prepare:
            q, k, v, gate, beta, z_silu = op.tensor_ir_op(
                create_gdn_prefill_prepare_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gdn_prefill_prepare",
                [qkv, z, alpha, beta_raw, self.A_log, self.dt_bias],
                [
                    Tensor.placeholder([b, s, n_kh, K], self.dtype),
                    Tensor.placeholder([b, s, n_kh, K], self.dtype),
                    Tensor.placeholder([b, s, n_vh, V], self.dtype),
                    Tensor.placeholder([b, s, n_vh], "float32"),
                    Tensor.placeholder([b, s, n_vh], "float32"),
                    Tensor.placeholder([b, s, n_vh * V], self.dtype),
                ],
            )
        else:
            # SiLU activation on QKV after conv
            qkv = op.silu(qkv)

            # Split QKV
            qkv_parts = op.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
            q = op.reshape(qkv_parts[0], (b, s, n_kh, K))
            k = op.reshape(qkv_parts[1], (b, s, n_kh, K))
            v = op.reshape(qkv_parts[2], (b, s, n_vh, V))

            # L2 normalize Q and K
            q = self._l2_normalize(q)
            k = self._l2_normalize(k)

            # Gate computation
            gate, beta = self._compute_gate_beta(alpha, beta_raw)
            z_silu = op.silu(z)
        # beta is already (b, s, n_vh) — no GVA expansion needed.

        if use_packed_decode_recurrent:
            pass
        elif use_direct_state_storage:
            if state_storage is None:
                state_storage = state.get_storage(
                    layer_idx,
                    0,
                    (n_vh, K, V),
                    "float32",
                    direct_state_storage_batch,
                    direct_state_max_history,
                )
            out_recurrent, _ = op.tensor_ir_inplace_op(
                create_gated_delta_net_decode_state_storage_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gated_delta_net_state_storage",
                [q, k, v, gate, beta, state_storage, seq_slot_ids, history_slot_ids],
                inplace_indices=[-1, 5],
                out=[
                    Tensor.placeholder([b, s, n_vh, V], "float32"),
                    state_storage,
                ],
            )
            state_out_layer = None
        elif use_chunked_prefill_recurrent:
            # The FLA/vLLM chunk formulation works with log-decay cumsums.
            # The existing model preparation produces exp(log_decay), so convert
            # back to log space here while leaving the stable default path alone.
            g_log = op.log(gate)
            g_cumsum = triton_ops.chunk_local_cumsum_scalar(
                g_log, chunk_size=chunked_gdn_chunk_size
            )
            a = triton_ops.chunk_scaled_dot_kkt(
                k,
                beta,
                g_cumsum,
                chunk_size=chunked_gdn_chunk_size,
                block_k=64,
            )
            ai = triton_ops.solve_tril_64(a, chunk_size=chunked_gdn_chunk_size)
            wu = triton_ops.recompute_w_u(
                k,
                v,
                beta,
                ai,
                g_cumsum,
                chunk_size=chunked_gdn_chunk_size,
                block_k=64,
                block_v=64,
            )
            w, u = op.split(wu, [K], axis=3)
            state_in_vk = op.permute_dims(state_in_layer, [0, 1, 3, 2])
            packed = triton_ops.chunk_delta_h_packed(
                k,
                w,
                u,
                g_cumsum,
                state_in_vk,
                chunk_size=chunked_gdn_chunk_size,
                block_v=64,
            )
            out_recurrent = triton_ops.chunk_fwd_o_packed(
                q,
                k,
                packed,
                g_cumsum,
                value_dim=V,
                chunk_size=chunked_gdn_chunk_size,
                block_k=64,
                block_v=64,
            )
            state_out_layer = triton_ops.chunk_delta_h_final_state(
                packed,
                s,
                value_dim=V,
                key_dim=K,
                chunk_size=chunked_gdn_chunk_size,
            )
        else:
            gdn_func = (
                create_gated_delta_net_decode_func
                if isinstance(s, int) and s == 1
                else create_gated_delta_net_func
            )

            # Recurrent computation via TIR kernel
            out_recurrent, state_out_layer = op.tensor_ir_op(
                gdn_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gated_delta_net",
                [q, k, v, gate, beta, state_in_layer],
                [
                    Tensor.placeholder([b, s, n_vh, V], "float32"),
                    Tensor.placeholder([b, n_vh, K, V], "float32"),
                ],
            )

        # Cast recurrent output back to model dtype
        out_recurrent = op.astype(out_recurrent, self.dtype)

        # Output gating
        out_normed = self.norm(out_recurrent)
        out_flat = op.reshape(out_normed, (b, s, n_vh * V))
        out_gated = out_flat * z_silu
        output = self.out_proj(out_gated)

        if new_conv_state is not None:
            state = state.set(layer_idx, 1, new_conv_state)
        if state_out_layer is not None:
            state = state.set(layer_idx, 0, state_out_layer)
        return output, state

    def _causal_conv1d_with_state(self, qkv: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        """Causal Conv1D using a pre-extracted conv_state tensor (for RNNState path)."""
        b, s, d = qkv.shape
        kernel_size = self.config.linear_conv_kernel_dim

        # Update conv state
        def _te_update_conv_state(old_state: te.Tensor, qkv_in: te.Tensor):
            ks_minus_1 = old_state.shape[1]
            seq = qkv_in.shape[1]
            return te.compute(
                old_state.shape,
                lambda bi, ti, di: tirx.if_then_else(
                    seq + ti < ks_minus_1,
                    old_state[bi, seq + ti, di],
                    qkv_in[bi, seq + ti - ks_minus_1, di],
                ),
                name="update_conv_state",
            )

        new_conv_state = op.tensor_expr_op(
            _te_update_conv_state, "update_conv_state", [conv_state, qkv]
        )

        # Depthwise conv
        def _te_depthwise_conv(state: te.Tensor, qkv_in: te.Tensor, weight: te.Tensor):
            ks_m1 = state.shape[1]
            seq = qkv_in.shape[1]
            kk = te.reduce_axis((0, kernel_size), name="kk")
            return te.compute(
                (qkv_in.shape[0], seq, qkv_in.shape[2]),
                lambda bi, si, di: te.sum(
                    tirx.if_then_else(
                        si + kk < ks_m1,
                        state[bi, si + kk, di],
                        qkv_in[bi, si + kk - ks_m1, di],
                    )
                    * weight[di, 0, kk],
                    axis=kk,
                ),
                name="depthwise_conv1d",
            )

        result = op.tensor_expr_op(
            _te_depthwise_conv,
            "depthwise_conv1d",
            [conv_state, qkv, self.conv1d_weight],
            attrs={"op_pattern": 8},
        )
        return result, new_conv_state

    def _l2_normalize(self, x: Tensor) -> Tensor:
        """L2 normalize along last dimension with eps=1e-6."""
        # x: (b, s, h, d) — compute in float32 for numerical stability
        x_f32 = op.astype(x, "float32")
        x_sq = x_f32 * x_f32
        sum_sq = op.sum(x_sq, axis=-1, keepdims=True)  # (b, s, h, 1)
        inv_norm = op.sqrt(sum_sq + 1e-6)
        return op.astype(x_f32 / inv_norm, self.dtype)

    def _compute_gate_beta(self, alpha: Tensor, beta_raw: Tensor):
        """Compute decay gate and update rate.

        gate = exp(-exp(A_log) * softplus(alpha + dt_bias))  (per value_head)
        beta = sigmoid(beta_raw)  (per value_head)
        """

        # alpha: (b, s, n_vh), dt_bias: (n_vh,), A_log: (n_vh,)
        def _te_gate(alpha: te.Tensor, A_log: te.Tensor, dt_bias: te.Tensor):
            b, s, h = alpha.shape

            def _softplus(x):
                # softplus(x) = x if x > 20 else log(1 + exp(x))
                return tirx.if_then_else(x > 20.0, x, tirx.log(1.0 + tirx.exp(x)))

            return te.compute(
                (b, s, h),
                lambda bi, si, hi: tirx.exp(
                    -tirx.exp(A_log[hi].astype("float32"))
                    * _softplus((alpha[bi, si, hi] + dt_bias[hi]).astype("float32"))
                ),
                name="gate",
            )

        gate = op.tensor_expr_op(
            _te_gate,
            "gate",
            [alpha, self.A_log, self.dt_bias],
            attrs={"op_pattern": 8},
        )

        beta = op.sigmoid(beta_raw).astype("float32")
        return gate, beta

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        # A_log and dt_bias must stay float32
        self.A_log.to("float32")
        self.dt_bias.to("float32")


# ============================================================================
# Decoder Layer (dispatches between GDN and standard attention)
# ============================================================================


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_id: int, category_id: int):
        """
        layer_id is the id of the layer within all of the layers
        category_id is the index of the layer within the category of layers that it belongs to
        ie, linear attention or regular attention
        """
        self.layer_type = config.layer_types()[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35GatedDeltaNet(config, category_id)
        self.category_id = category_id
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # noqa: UP006
        seq_slot_ids: Optional[Tensor] = None,
        history_slot_ids: Optional[Tensor] = None,
        state_storage: Optional[Tensor] = None,
        conv_state_storage: Optional[Tensor] = None,
    ):
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, self.category_id, position_embeddings)
        else:
            out, state = self.linear_attn.forward(
                out,
                state,
                seq_slot_ids,
                history_slot_ids,
                state_storage,
                conv_state_storage,
            )
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states, state

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.config = config
        self.embed_tokens = Qwen35Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = None
        if config.mrope_section is not None:
            self.rotary_emb = MultimodalRotaryEmbedding(
                head_dim=int(config.head_dim * config.partial_rotary_factor),
                theta=config.rope_theta,
                mrope_section=config.mrope_section,
                attention_scaling=1.0,
                interleaved=config.mrope_interleaved,
            )
        layer_types = config.layer_types()
        linear_idx = 0
        attn_idx = 0
        layers = []
        for i, ltype in enumerate(layer_types):
            if ltype == "linear_attention":
                layers.append(Qwen35DecoderLayer(config, i, category_id=linear_idx))
                linear_idx += 1
            else:
                layers.append(Qwen35DecoderLayer(config, i, category_id=attn_idx))
                attn_idx += 1
        self.num_linear_layers = linear_idx
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        position_ids: Optional[Tensor] = None,
    ):
        hidden_states = inputs
        batch_size, seq_len, _ = hidden_states.shape
        use_direct_state_storage = _allow_direct_rnn_state_for_decode(
            self.config, batch_size, seq_len
        )
        if use_direct_state_storage:
            seq_slot_ids = state.get_seq_slot_ids(batch_size)
            history_slot_ids = state.get_history_slot_ids(batch_size)
            direct_state_max_history = int(
                os.environ.get("MLC_QWEN35_DIRECT_RNN_MAX_HISTORY", "1")
            )
            direct_state_storage_batch = int(
                os.environ.get(
                    "MLC_QWEN35_DIRECT_RNN_STORAGE_BATCH",
                    str(self.config.max_batch_size),
                )
            )
            use_batch_storage = os.environ.get("MLC_QWEN35_BATCH_RNN_STORAGE", "0") == "1"
            if use_batch_storage:
                state_storages = state.get_storages(
                    self.num_linear_layers,
                    0,
                    (
                        self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim,
                        self.config.linear_value_head_dim,
                    ),
                    "float32",
                    direct_state_storage_batch,
                    direct_state_max_history,
                )
            else:
                state_storages = None
            if (
                use_batch_storage
                and os.environ.get("MLC_QWEN35_DIRECT_CONV_STATE", "0") == "1"
            ):
                direct_conv_state_dtype = _direct_conv_state_dtype(self.config.dtype)
                qkv_dim = (
                    self.config.linear_num_key_heads * self.config.linear_key_head_dim * 2
                    + self.config.linear_num_value_heads * self.config.linear_value_head_dim
                )
                conv_state_storages = state.get_storages(
                    self.num_linear_layers,
                    1,
                    (self.config.linear_conv_kernel_dim - 1, qkv_dim),
                    direct_conv_state_dtype,
                    direct_state_storage_batch,
                    1,
                )
            else:
                conv_state_storages = None
        else:
            seq_slot_ids = None
            history_slot_ids = None
            state_storages = None
            conv_state_storages = None
        if self.rotary_emb is not None and position_ids is None:
            base = paged_kv_cache.get_query_positions(seq_len)
            base = op.reshape(base, (1, seq_len))
            base = op.broadcast_to(base, (batch_size, seq_len))
            base = op.unsqueeze(base, dim=0)
            position_ids = op.broadcast_to(base, (3, batch_size, seq_len))
        position_embeddings = (
            None
            if self.rotary_emb is None or position_ids is None
            else self.rotary_emb(hidden_states, position_ids)
        )
        for layer_id, layer in enumerate(self.layers):
            state_storage = None
            conv_state_storage = None
            if layer.layer_type == "linear_attention" and state_storages is not None:
                state_storage = state_storages[layer.category_id]
                if conv_state_storages is not None:
                    conv_state_storage = conv_state_storages[layer.category_id]
            hidden_states, state = layer.forward(
                hidden_states,
                paged_kv_cache,
                state,
                position_embeddings,
                seq_slot_ids,
                history_slot_ids,
                state_storage,
                conv_state_storage,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, state


class Qwen35LMHeadModel(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.config = config
        self.model = Qwen35Model(config)
        self.visual = Qwen35VisionModel(config) if config.vision_config else None
        if self.visual is not None:
            # Keep image encoding dense under weight-only quantization. Decode speed
            # is dominated by the language model, while quantizing the vision tower
            # is a larger correctness risk for image understanding.
            self.visual.no_quantization = True
        self.image_processor = ImageProcessor()
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        self.image_dtype = "uint8"
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.partial_rotary_factor = config.partial_rotary_factor
        # GDN config
        self.num_linear_layers = config.num_linear_layers
        self.num_attention_layers = config.num_attention_layers
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def image_preprocess(
        self,
        pixel_values: Tensor,
        resized_height: int,
        resized_width: int,
    ) -> Tensor:
        pixel_values = op.permute_dims(pixel_values, axes=(0, 3, 1, 2))
        pixel_values = self.image_processor.resize(
            pixel_values, params={"height": resized_height, "width": resized_width}
        )
        pixel_values = self.image_processor.rescale(pixel_values, o_dtype=self.dtype)
        pixel_values = pixel_values * 2.0 - 1.0
        pixel_values = op.unsqueeze(pixel_values, dim=2)
        if self.config.vision_config.get("temporal_patch_size", 2) == 2:
            pixel_values = op.concat([pixel_values, pixel_values], dim=2)
        return pixel_values

    def image_embed(
        self,
        pixel_values: Tensor,
        resized_height: int,
        resized_width: int,
        crop_height: int,
        crop_width: int,
    ) -> Tensor:
        if self.visual is None:
            raise ValueError("Qwen3.5 image embedding requires `vision_config`.")
        pixel_values = self.image_preprocess(pixel_values, resized_height, resized_width)
        return self.visual(pixel_values, crop_height, crop_width)

    def get_logits(self, hidden_states: Tensor):
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def get_token_ids(self, hidden_states: Tensor):
        """Greedy-only fused LM-head top-1 helper.

        This is exported only when explicitly requested and is used by the
        serving fast path after it verifies that logits processors are inactive.
        """
        b, s, h = hidden_states.shape
        hidden_2d = op.reshape(hidden_states, (b * s, h))
        if self.tie_word_embeddings:
            weight = self.model.embed_tokens.weight
        else:
            weight = self.lm_head.weight
        block_m = int(os.environ.get("MLC_QWEN35_LM_HEAD_ARGMAX_BLOCK_M", "32"))
        block_k = int(os.environ.get("MLC_QWEN35_LM_HEAD_ARGMAX_BLOCK_K", "1024"))
        return triton_ops.lm_head_argmax(hidden_2d, weight, block_m=block_m, block_k=block_k)

    def _forward_hidden(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        """Shared forward for batch methods using RNNState."""
        op_ext.configure()
        hidden_states, state = self.model.forward(input_embed, paged_kv_cache, state, position_ids)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return hidden_states, paged_kv_cache, state

    def _forward(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        hidden_states, paged_kv_cache, state = self._forward_hidden(
            input_embed, paged_kv_cache, state, logit_positions, position_ids
        )
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache, state

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, logit_positions)

    def batch_prefill_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_hidden(input_embeds, paged_kv_cache, rnn_state, logit_positions)

    def _set_mrope_delta(self, paged_kv_cache: PagedKVCache, deltas: Tensor):
        setattr(paged_kv_cache, "_mrope_delta", deltas)
        return deltas

    def _get_mrope_delta(self, paged_kv_cache: PagedKVCache) -> Optional[Tensor]:
        return getattr(paged_kv_cache, "_mrope_delta", None)

    def _build_decode_position_ids(
        self,
        seq_len: int,
        paged_kv_cache: PagedKVCache,
        batch: int,
    ) -> Optional[Tensor]:
        delta = self._get_mrope_delta(paged_kv_cache)
        return self._build_decode_position_ids_from_query_positions(
            paged_kv_cache.get_query_positions(seq_len),
            seq_len,
            batch,
            delta,
        )

    def _build_decode_position_ids_from_query_positions(
        self,
        query_positions: Tensor,
        seq_len: int,
        batch: int,
        delta: Optional[Tensor],
    ) -> Tensor:
        base = query_positions
        base = op.reshape(base, (1, seq_len))
        base = op.broadcast_to(base, (batch, seq_len))
        if delta is not None:
            base = base + delta
        base = op.unsqueeze(base, dim=0)
        return op.broadcast_to(base, (3, batch, seq_len))

    def batch_prefill_mrope(
        self,
        input_embeds: Tensor,
        position_ids: Tensor,
        mrope_deltas: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        return self._forward(
            input_embeds,
            paged_kv_cache,
            rnn_state,
            logit_positions,
            position_ids,
        )

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        b, s, _ = input_embeds.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b)
        return self._forward(input_embeds, paged_kv_cache, rnn_state, position_ids=position_ids)

    def batch_decode_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        b, s, _ = input_embeds.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b)
        return self._forward_hidden(input_embeds, paged_kv_cache, rnn_state, position_ids=position_ids)

    def decode(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        position_ids = self._build_decode_position_ids(1, paged_kv_cache, 1)
        return self._forward(input_embed, paged_kv_cache, rnn_state, position_ids=position_ids)

    def decode_to_last_hidden_states(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        position_ids = self._build_decode_position_ids(1, paged_kv_cache, 1)
        return self._forward_hidden(input_embed, paged_kv_cache, rnn_state, position_ids=position_ids)

    def batch_decode_mrope(
        self,
        input_embeds: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        return self.batch_decode(input_embeds, paged_kv_cache, rnn_state)

    def batch_decode_mrope_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        return self.batch_decode_to_last_hidden_states(input_embeds, paged_kv_cache, rnn_state)

    def decode_mrope(
        self,
        input_embed: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        return self.decode(input_embed, paged_kv_cache, rnn_state)

    def decode_mrope_with_query_positions(
        self,
        input_embed: Tensor,
        mrope_deltas: Tensor,
        query_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        position_ids = self._build_decode_position_ids_from_query_positions(
            query_positions,
            seq_len=1,
            batch=1,
            delta=mrope_deltas,
        )
        return self._forward(input_embed, paged_kv_cache, rnn_state, position_ids=position_ids)

    def decode_mrope_to_last_hidden_states(
        self,
        input_embed: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        return self.decode_to_last_hidden_states(input_embed, paged_kv_cache, rnn_state)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        b, s, _ = input_embeds.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b)
        return self._forward(input_embeds, paged_kv_cache, rnn_state, position_ids=position_ids)

    def batch_verify_mrope(
        self,
        input_embeds: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        # Verification can pack multiple request trees into a single
        # [1, total_tokens, hidden] tensor, so C++ passes per-token deltas as
        # [1, total_tokens] instead of per-sequence [batch, 1].
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        return self.batch_verify(input_embeds, paged_kv_cache, rnn_state)

    def create_rnn_state(
        self,
        max_batch_size: tirx.Var,
        max_history: tirx.Var,
    ) -> RNNState:
        K = self.linear_key_head_dim
        V = self.linear_value_head_dim
        n_vh = self.linear_num_value_heads
        n_kh = self.config.linear_num_key_heads
        qkv_dim = n_kh * K * 2 + n_vh * V
        conv_ks_m1 = self.config.linear_conv_kernel_dim - 1

        init_values = [
            R.const(np.zeros((n_vh, K, V), "float32")),
            R.const(np.zeros((conv_ks_m1, qkv_dim), self.dtype)),
        ]

        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_linear_layers,
            max_history=max_history,
            init_values=init_values,
        )

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        forced_page_size = os.environ.get("MLC_QWEN35_KV_PAGE_SIZE")
        if forced_page_size:
            page_size_value = int(forced_page_size)
            if page_size_value <= 0:
                raise ValueError("MLC_QWEN35_KV_PAGE_SIZE must be positive when set.")
            page_size = tirx.IntImm("int64", page_size_value)
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            # Only attention layers use the KV cache
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rotary_dim=rotary_dim,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_embed": {
                "pixel_values": nn.spec.Tensor(
                    [1, "image_height", "image_width", 3], self.image_dtype
                ),
                "resized_height": nn.spec.Int(),
                "resized_width": nn.spec.Int(),
                "crop_height": nn.spec.Int(),
                "crop_width": nn.spec.Int(),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_mrope": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "position_ids": nn.spec.Tensor([3, 1, "seq_len"], "int32"),
                "mrope_deltas": nn.spec.Tensor(["batch_size", 1], "int32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode_mrope": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor(["batch_size", 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode_mrope": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify_mrope": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor([1, "seq_len"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        if os.environ.get("MLC_QWEN35_OMIT_VERIFY", "0") == "1":
            mod_spec.pop("batch_verify", None)
            mod_spec.pop("batch_verify_mrope", None)
        if self.visual is None:
            mod_spec.pop("image_embed")
        if os.environ.get("MLC_QWEN35_EXPORT_DECODE_MROPE_QUERY_POSITIONS", "0") == "1":
            mod_spec["decode_mrope_with_query_positions"] = {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                "query_positions": nn.spec.Tensor([1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            }
        enable_hidden_funcs = os.environ.get("MLC_QWEN35_ENABLE_HIDDEN_FUNCS", "0") == "1"
        enable_probes = os.environ.get("MLC_QWEN35_ENABLE_PROBES", "0") == "1"
        if enable_hidden_funcs or enable_probes:
            mod_spec.update(
                {
                    "batch_prefill_to_last_hidden_states": {
                        "input_embeds": nn.spec.Tensor(
                            [1, "seq_len", self.hidden_size], self.dtype
                        ),
                        "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                        "rnn_state": nn.spec.Object(object_type=RNNState),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                    "batch_decode_to_last_hidden_states": {
                        "input_embeds": nn.spec.Tensor(
                            ["batch_size", 1, self.hidden_size], self.dtype
                        ),
                        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                        "rnn_state": nn.spec.Object(object_type=RNNState),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                    "decode_to_last_hidden_states": {
                        "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                        "rnn_state": nn.spec.Object(object_type=RNNState),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                    "batch_decode_mrope_to_last_hidden_states": {
                        "input_embeds": nn.spec.Tensor(
                            ["batch_size", 1, self.hidden_size], self.dtype
                        ),
                        "mrope_deltas": nn.spec.Tensor(["batch_size", 1], "int32"),
                        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                        "rnn_state": nn.spec.Object(object_type=RNNState),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                    "decode_mrope_to_last_hidden_states": {
                        "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                        "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                        "rnn_state": nn.spec.Object(object_type=RNNState),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                }
            )
        if enable_probes:
            mod_spec.update(
                {
                    "get_logits": {
                        "hidden_states": nn.spec.Tensor(
                            ["batch_size", "seq_len", self.hidden_size], self.dtype
                        ),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                }
            )
        if os.environ.get("MLC_QWEN35_FUSED_LM_HEAD_ARGMAX", "0") == "1":
            mod_spec.update(
                {
                    "get_token_ids": {
                        "hidden_states": nn.spec.Tensor(
                            ["batch_size", "seq_len", self.hidden_size], self.dtype
                        ),
                        "$": {
                            "param_mode": "packed",
                            "effect_mode": "none",
                        },
                    },
                }
            )
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
