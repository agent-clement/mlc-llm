"""Attention KV cache modeling."""

import json
import os
from typing import Any, Dict, List, Literal, Optional, Union  # noqa: UP035

import numpy as np
from tvm import relax as rx
from tvm import tirx
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache as TVMPagedKVCache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode


class PagedKVCache(TVMPagedKVCache):
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    def append_mha_kv_cross_attention(
        self,
        layer_id: int,
        q: Any,
        k: Any,
        v: Any,
        v_head_dim: int,
        sm_scale: float,
    ) -> tuple[Any, Any]:
        """Append MHA K/V data to KV cache and compute paged cross attention."""
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.struct_info.shape
        _, _, h_kv, _ = k._expr.struct_info.shape
        _, _, _, d_v = v._expr.struct_info.shape
        q = q.reshape(b * s, h_qo, d_qk)
        k = k.reshape(b * s, h_kv, d_qk)
        v = v.reshape(b * s, h_kv, d_v)
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.attention_kv_cache_append_mha_kv_cross_attention",
                [
                    self._expr,
                    rx.PrimValue(layer_id),  # type: ignore[arg-type]
                    rx.PrimValue(sm_scale),
                    q._expr,
                    k._expr,
                    v._expr,
                ],
                out_sinfo=[
                    rx.TensorStructInfo((b * s, h_qo, v_head_dim), q.dtype),
                    rx.TensorStructInfo((b * s, h_qo), "float32"),
                ],
            )
        )
        assert isinstance(attn_results.struct_info, rx.TupleStructInfo)
        assert len(attn_results.struct_info.fields) == 2
        o = PagedKVCache._wrap_tensor(bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(
            b, s, h_qo, v_head_dim
        )
        lse = PagedKVCache._wrap_tensor(bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(
            b, s, h_qo
        )
        return o, lse

    @staticmethod
    def _wrap_tensor(expr: Any) -> Any:
        # Imported lazily to avoid depending on the exact frontend tensor class path.
        from tvm.relax.frontend.nn import Tensor  # pylint: disable=import-outside-toplevel

        return Tensor(_expr=expr)

    def append_mha_kv(self, layer_id: int, k: Any, v: Any) -> "PagedKVCache":
        """Fine-grained API that appends MHA K/V data to KV cache."""
        # pylint: disable=protected-access
        b, s, h_kv, d_qk = k._expr.struct_info.shape
        _, _, _, d_v = v._expr.struct_info.shape
        k = k.reshape(b * s, h_kv, d_qk)
        v = v.reshape(b * s, h_kv, d_v)
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_append_mha_kv",
                self._expr,
                rx.PrimValue(layer_id),  # type: ignore[arg-type]
                k._expr,
                v._expr,
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name="paged_kv_cache",
        )

    def get_paged_decode_metadata(self, layer_id: int, depth: int = 0) -> Any:
        """Return explicit paged decode metadata for the current BeginForward window."""
        bb = rx.BlockBuilder.current()
        metadata = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_get_paged_decode_metadata",
                self._expr,
                rx.PrimValue(layer_id),  # type: ignore[arg-type]
                rx.PrimValue(depth),
                sinfo_args=rx.ObjectStructInfo(),
            )
        )
        return metadata

    def get_paged_decode_metadata_tensors(
        self,
        layer_id: int,
        depth: int = 0,
        kv_dtype: str = "float16",
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return typed MLC paged decode metadata tensors.

        This is the compiler-visible form of :meth:`get_paged_decode_metadata`.
        It keeps the cache pages, CSR page metadata, length info, and MRoPE
        position maps as ordinary Relax tensor expressions.
        """
        bb = rx.BlockBuilder.current()
        metadata = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_get_paged_decode_metadata",
                self._expr,
                rx.PrimValue(layer_id),  # type: ignore[arg-type]
                rx.PrimValue(depth),
                sinfo_args=rx.TupleStructInfo(
                    [
                        rx.TensorStructInfo(ndim=5, dtype=kv_dtype),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                    ]
                ),
            )
        )
        assert isinstance(metadata.struct_info, rx.TupleStructInfo)
        return tuple(bb.emit(rx.TupleGetItem(metadata, i)) for i in range(6))

    def get_fa2_paged_decode_metadata(self, layer_id: int, depth: int = 0) -> Any:
        """Return FA2/vLLM-style batch-1 paged decode K/V views and metadata."""
        bb = rx.BlockBuilder.current()
        metadata = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_get_fa2_paged_decode_metadata",
                self._expr,
                rx.PrimValue(layer_id),  # type: ignore[arg-type]
                rx.PrimValue(depth),
                sinfo_args=rx.ObjectStructInfo(),
            )
        )
        return metadata

    def get_fa2_paged_decode_metadata_tensors(
        self, layer_id: int, depth: int = 0, kv_dtype: str = "float16"
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return typed FA2/vLLM-style batch-1 paged decode metadata tensors.

        This is the compiler-visible form of :meth:`get_fa2_paged_decode_metadata`.
        It keeps the paged K/V views, block table, sequence lengths, and MRoPE
        position map as ordinary Relax tensor expressions so a future FA2 paged
        decode call can consume them without reading hidden KV-cache object state.
        """
        bb = rx.BlockBuilder.current()
        use_vllm_cache_layout = os.environ.get("MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT", "0") == "1"
        k_pages_sinfo = (
            rx.TensorStructInfo(ndim=5, dtype=kv_dtype)
            if use_vllm_cache_layout
            else rx.TensorStructInfo(ndim=4, dtype=kv_dtype)
        )
        metadata = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_get_fa2_paged_decode_metadata",
                self._expr,
                rx.PrimValue(layer_id),  # type: ignore[arg-type]
                rx.PrimValue(depth),
                sinfo_args=rx.TupleStructInfo(
                    [
                        k_pages_sinfo,
                        rx.TensorStructInfo(ndim=4, dtype=kv_dtype),
                        rx.TensorStructInfo(ndim=2, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                        rx.TensorStructInfo(ndim=1, dtype="int32"),
                    ]
                ),
            )
        )
        assert isinstance(metadata.struct_info, rx.TupleStructInfo)
        return tuple(bb.emit(rx.TupleGetItem(metadata, i)) for i in range(6))

    def fa2_paged_decode_attention(
        self,
        layer_id: int,
        q: Any,
        v_head_dim: int,
        sm_scale: float,
        depth: int = 0,
    ) -> tuple[Any, Any]:
        """Emit an explicit FA2 paged decode call using typed metadata tensors.

        This is the compiler-facing integration point for a future paged
        FlashAttention backend. Unlike the packed KV-cache cross-attention call,
        all paged metadata is carried as normal tensor arguments.
        """
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.struct_info.shape
        q_flat = q.reshape(b * s, h_qo, d_qk)
        k_pages, v_pages, block_table, seqused_k, cu_seqlens_q, q_rope_position = (
            self.get_fa2_paged_decode_metadata_tensors(layer_id, depth, q.dtype)
        )
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.attention_kv_cache_fa2_paged_decode",
                [
                    q_flat._expr,
                    k_pages,
                    v_pages,
                    block_table,
                    seqused_k,
                    cu_seqlens_q,
                    q_rope_position,
                    rx.PrimValue(layer_id),  # type: ignore[arg-type]
                    rx.PrimValue(depth),
                    rx.PrimValue(sm_scale),
                ],
                out_sinfo=[
                    rx.TensorStructInfo((b * s, h_qo, v_head_dim), q.dtype),
                    rx.TensorStructInfo((b * s, h_qo), "float32"),
                ],
            )
        )
        assert isinstance(attn_results.struct_info, rx.TupleStructInfo)
        assert len(attn_results.struct_info.fields) == 2
        o = PagedKVCache._wrap_tensor(bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(
            b, s, h_qo, v_head_dim
        )
        lse = PagedKVCache._wrap_tensor(bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(
            b, s, h_qo
        )
        return o, lse

    def cross_attention_with_paged_metadata(
        self,
        layer_id: int,
        q: Any,
        v_head_dim: int,
        sm_scale: float,
        depth: int = 0,
        metadata: Optional[tuple[Any, Any, Any, Any, Any, Any]] = None,
    ) -> tuple[Any, Any]:
        """Compute paged cross attention with compiler-visible metadata tensors."""
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.struct_info.shape
        q_flat = q.reshape(b * s, h_qo, d_qk)
        if metadata is None:
            metadata = self.get_paged_decode_metadata_tensors(layer_id, depth, q.dtype)
        pages, page_indptr, page_indices, length_info, k_rope_pos, q_rope_position = metadata
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.attention_kv_cache_cross_attention_with_paged_metadata",
                [
                    self._expr,
                    rx.PrimValue(layer_id),  # type: ignore[arg-type]
                    rx.PrimValue(depth),
                    rx.PrimValue(sm_scale),
                    q_flat._expr,
                    pages,
                    page_indptr,
                    page_indices,
                    length_info,
                    k_rope_pos,
                    q_rope_position,
                ],
                out_sinfo=[
                    rx.TensorStructInfo((b * s, h_qo, v_head_dim), q.dtype),
                    rx.TensorStructInfo((b * s, h_qo), "float32"),
                ],
            )
        )
        assert isinstance(attn_results.struct_info, rx.TupleStructInfo)
        assert len(attn_results.struct_info.fields) == 2
        o = PagedKVCache._wrap_tensor(bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(
            b, s, h_qo, v_head_dim
        )
        lse = PagedKVCache._wrap_tensor(bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(
            b, s, h_qo
        )
        return o, lse

    @staticmethod
    def create_generic(
        attn_kind: Union[Literal["mha", "mla"], List[Literal["mha", "mla", "mha_sliding"]]],  # noqa: UP006
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        mla_original_qk_head_dim: int = 0,
        mla_original_v_head_dim: int = 0,
        rotary_dim: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,  # noqa: UP006
        rope_ext_factors: Optional[List[int]] = None,  # noqa: UP006
        layer_partition: Optional[List[int]] = None,  # noqa: UP006
        enable_disaggregation: bool = False,
        name: str = "paged_kv_cache",
    ) -> "PagedKVCache":
        """The generic function of creating a multi-head attention PagedKVCache,
        which will be rewritten by functions in compilation pipeline.
        """
        if rotary_dim is None:
            rotary_dim = qk_head_dim
        if rope_scaling is None:
            rope_scaling = {}
        if layer_partition is None:
            layer_partition = [0, num_hidden_layers]
        if isinstance(attn_kind, List):  # noqa: UP006
            rx_attn_kind = [rx.StringImm(layer_kind) for layer_kind in attn_kind]
        else:
            rx_attn_kind = rx.StringImm(attn_kind)
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "mlc.create_paged_kv_cache_generic",
                rx_attn_kind,
                rx.ShapeExpr(
                    [
                        max_batch_size,
                        max_total_seq_len,
                        prefill_chunk_size,
                        page_size,
                        support_sliding_window,
                    ]
                ),
                rx.ShapeExpr(layer_partition),
                rx.PrimValue(num_hidden_layers),
                rx.PrimValue(num_attention_heads),
                rx.PrimValue(num_key_value_heads),
                rx.PrimValue(qk_head_dim),
                rx.PrimValue(v_head_dim),
                rx.PrimValue(mla_original_qk_head_dim),
                rx.PrimValue(mla_original_v_head_dim),
                rx.PrimValue(rope_mode),
                rx.PrimValue(rope_scale),
                rx.PrimValue(rope_theta),
                rx.StringImm(json.dumps(rope_scaling)),
                (
                    rx.const(np.array(rope_ext_factors, "float32"))
                    if rope_ext_factors is not None
                    else rx.PrimValue(0)
                    # NOTE: since relax does not have "Optional" type, we use PrimValue(0)
                    # to represent "undefined".
                ),
                rx.PrimValue(rotary_dim),
                rx.PrimValue(int(enable_disaggregation)),
                rx.DataTypeImm(dtype),
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )
