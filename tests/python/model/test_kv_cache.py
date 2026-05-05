import numpy as np
import pytest
import tvm
from tvm import relax, tirx
from tvm.relax.frontend.nn import core, modules, spec
from tvm.relax.frontend.nn.llm import kv_cache as tvm_kv_cache
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

from mlc_llm.nn.kv_cache import PagedKVCache, RopeMode

# mypy: disable-error-code="attr-defined"


def test_nn_module_paged_kv_cache():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def create_paged_kv_cache(
            max_batch_size_1: R.Shape(["max_batch_size"]),
            max_total_seq_len_1: R.Shape(["max_total_seq_len"]),
            prefill_chunk_size_1: R.Shape(["prefill_chunk_size"]),
            page_size_1: R.Shape(["page_size"]),
            support_sliding_window_1: R.Shape(["support_sliding_window"]),
        ) -> R.Object:
            max_batch_size = T.int64()
            max_total_seq_len = T.int64()
            prefill_chunk_size = T.int64()
            page_size = T.int64()
            support_sliding_window = T.int64()
            R.func_attr({"num_input": 5})
            with R.dataflow():
                paged_kv_cache: R.Object = R.call_pure_packed("mlc.create_paged_kv_cache_generic", R.str("mha"), R.shape([max_batch_size, max_total_seq_len, prefill_chunk_size, page_size, support_sliding_window]), R.shape([0, 32]), R.prim_value(32), R.prim_value(32), R.prim_value(32), R.prim_value(128), R.prim_value(128), R.prim_value(0), R.prim_value(0), R.prim_value(1), R.prim_value(1), R.prim_value(10000), R.str("{}"), R.prim_value(0), R.prim_value(128), R.prim_value(0), R.dtype("float16"), sinfo_args=(R.Object,))  # noqa: E501
                gv1: R.Object = paged_kv_cache
                R.output(gv1)
            return gv1

        @R.function
        def forward(
            cache: R.Object, qkv: R.Tensor((1, 100, 96, 128), dtype="float16")
        ) -> R.Tensor((1, 100, 32, 128), dtype="float16"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                reshape: R.Tensor((100, 96, 128), dtype="float16") = R.reshape(
                    qkv, R.shape([100, 96, 128])
                )
                lv = R.call_dps_packed(
                    "vm.builtin.attention_kv_cache_attention_with_fused_qkv",
                    (cache, R.prim_value(0), R.prim_value(T.float32(0.08838834764831845)), reshape),
                    out_sinfo=R.Tensor((100, 32, 128), dtype="float16"),
                )
                reshape1: R.Tensor((1, 100, 32, 128), dtype="float16") = R.reshape(
                    lv, R.shape([1, 100, 32, 128])
                )
                gv: R.Tensor((1, 100, 32, 128), dtype="float16") = reshape1
                R.output(gv)
            return gv
    # fmt: on

    class PagedKVCacheTest(modules.Module):
        def forward(
            self,
            cache: PagedKVCache,
            qkv: core.Tensor,
        ) -> core.Tensor:
            return cache.attention_with_fused_qkv(0, qkv, num_qo_heads=32, sm_scale=128**-0.5)

        def create_paged_kv_cache(
            self,
            max_batch_size: tirx.Var,
            max_total_seq_len: tirx.Var,
            prefill_chunk_size: tirx.Var,
            page_size: tirx.Var,
            support_sliding_window: tirx.Var,
        ) -> PagedKVCache:
            return PagedKVCache.create_generic(
                attn_kind="mha",
                max_batch_size=max_batch_size,
                max_total_seq_len=max_total_seq_len,
                prefill_chunk_size=prefill_chunk_size,
                page_size=page_size,
                support_sliding_window=support_sliding_window,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                qk_head_dim=128,
                v_head_dim=128,
                rope_mode=RopeMode.NORMAL,
                rope_scale=1,
                rope_theta=10000,
                rotary_dim=128,
                dtype="float16",
            )

    export_results = PagedKVCacheTest().export_tvm(
        spec={
            "forward": {
                "cache": spec.Object(object_type=PagedKVCache),
                "qkv": spec.Tensor((1, 100, 96, 128), "float16"),
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
            },
        },
    )
    tvm_mod = export_results[0]
    tvm.ir.assert_structural_equal(tvm_mod, Module, True)


def test_nn_module_fa2_paged_decode_metadata_tensors():
    cache_var = relax.Var("cache", relax.ObjectStructInfo())
    bb = relax.BlockBuilder()
    with bb.function("forward", params=[cache_var]):
        cache = PagedKVCache(_expr=cache_var, _name="cache")
        _, _, block_table, _, _, _ = cache.get_fa2_paged_decode_metadata_tensors(
            0, kv_dtype="float16"
        )
        bb.emit_func_output(block_table)
    tvm_mod = bb.finalize()
    calls = []

    def visit(expr):
        if (
            isinstance(expr, relax.Call)
            and expr.op == tvm.ir.Op.get("relax.call_pure_packed")
            and isinstance(expr.args[0], relax.ExternFunc)
            and expr.args[0].global_symbol
            == "vm.builtin.attention_kv_cache_get_fa2_paged_decode_metadata"
        ):
            calls.append(expr)

    relax.analysis.post_order_visit(tvm_mod["forward"], visit)
    assert len(calls) == 1
    metadata_sinfo = calls[0].struct_info
    assert isinstance(metadata_sinfo, relax.TupleStructInfo)
    assert len(metadata_sinfo.fields) == 6
    expected = [
        (4, "float16"),
        (4, "float16"),
        (2, "int32"),
        (1, "int32"),
        (1, "int32"),
        (1, "int32"),
    ]
    for field, (ndim, dtype) in zip(metadata_sinfo.fields, expected):
        assert isinstance(field, relax.TensorStructInfo)
        assert field.ndim == ndim
        assert field.dtype == dtype


def test_nn_module_fa2_paged_decode_uses_vllm_cache_layout_sinfo(monkeypatch):
    monkeypatch.setenv("MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT", "1")
    cache_var = relax.Var("cache", relax.ObjectStructInfo())
    bb = relax.BlockBuilder()
    with bb.function("forward", params=[cache_var]):
        cache = PagedKVCache(_expr=cache_var, _name="cache")
        k_pages, _, _, _, _, _ = cache.get_fa2_paged_decode_metadata_tensors(
            0, kv_dtype="float16"
        )
        bb.emit_func_output(k_pages)
    tvm_mod = bb.finalize()
    calls = []

    def visit(expr):
        if (
            isinstance(expr, relax.Call)
            and expr.op == tvm.ir.Op.get("relax.call_pure_packed")
            and isinstance(expr.args[0], relax.ExternFunc)
            and expr.args[0].global_symbol
            == "vm.builtin.attention_kv_cache_get_fa2_paged_decode_metadata"
        ):
            calls.append(expr)

    relax.analysis.post_order_visit(tvm_mod["forward"], visit)
    assert len(calls) == 1
    metadata_sinfo = calls[0].struct_info
    assert isinstance(metadata_sinfo, relax.TupleStructInfo)
    k_pages_sinfo = metadata_sinfo.fields[0]
    assert isinstance(k_pages_sinfo, relax.TensorStructInfo)
    assert k_pages_sinfo.ndim == 5
    assert k_pages_sinfo.dtype == "float16"


def test_vllm_cache_layout_selects_matching_page_helpers(monkeypatch):
    monkeypatch.setenv("MLC_QWEN35_FA2_VLLM_CACHE_LAYOUT", "1")
    target = tvm.target.Target("cuda")

    copy = tvm_kv_cache._select_copy_single_page(2, 16, 256, "float16", target)
    copy_cpu = tvm_kv_cache._select_copy_single_page_cpu(2, 16, 256, "float16")
    debug = tvm_kv_cache._select_kv_cache_debug_get_kv(24, 2, 256, "float16")
    compact = tvm_kv_cache._select_compact_kv_copy(2, 256, "float16", target)
    compact_cpu = tvm_kv_cache._select_compact_kv_copy_cpu(2, 256, "float16")

    assert "copy_vllm" in copy.script()
    assert "copy_vllm" in copy_cpu.script()
    assert "copy0_vllm" in debug.script()
    assert "compact_kv_copy_vllm_layout" in compact.script()
    assert "compact_kv_copy_cpu_vllm_layout" in compact_cpu.script()


def test_vllm_cache_layout_page_helpers_preserve_logical_kv():
    num_layers = 1
    num_heads = 2
    head_dim = 8
    page_size = 4
    num_pages = 3
    seq_len = 8
    target = "llvm"

    append = tvm.tirx.build(
        tvm_kv_cache._kv_cache_transpose_append_vllm_layout(
            num_heads, head_dim, "float16", page_size
        ),
        target=target,
    )
    debug = tvm.tirx.build(
        tvm_kv_cache._kv_cache_debug_get_kv_vllm_layout(
            num_layers, num_heads, head_dim, "float16", page_size
        ),
        target=target,
    )
    copy_page = tvm.tirx.build(
        tvm_kv_cache._copy_single_page_cpu_vllm_layout(
            num_heads, page_size, head_dim, "float16"
        ),
        target=target,
    )
    compact = tvm.tirx.build(
        tvm_kv_cache._compact_kv_copy_cpu_vllm_layout(
            num_heads, head_dim, "float16", page_size
        ),
        target=target,
    )

    k_np = (
        np.arange(seq_len * num_heads * head_dim).reshape(seq_len, num_heads, head_dim) + 1
    ).astype("float16")
    v_np = (k_np + 1000).astype("float16")
    positions_np = np.arange(seq_len, dtype="int32")
    pages = tvm.runtime.tensor(
        np.zeros((num_pages, 2, num_heads, page_size, head_dim), dtype="float16")
    )
    positions = tvm.runtime.tensor(positions_np)
    append(pages, tvm.runtime.tensor(k_np), tvm.runtime.tensor(v_np), positions)

    k_actual = tvm.runtime.empty((num_layers, seq_len, num_heads, head_dim), "float16")
    v_actual = tvm.runtime.empty((num_layers, seq_len, num_heads, head_dim), "float16")
    debug(pages, positions, k_actual, v_actual, 0)
    np.testing.assert_array_equal(k_actual.numpy()[0], k_np)
    np.testing.assert_array_equal(v_actual.numpy()[0], v_np)

    copy_page(pages, 0, 2, page_size)
    copied_positions = tvm.runtime.tensor(np.arange(2 * page_size, 3 * page_size, dtype="int32"))
    k_copied = tvm.runtime.empty((num_layers, page_size, num_heads, head_dim), "float16")
    v_copied = tvm.runtime.empty((num_layers, page_size, num_heads, head_dim), "float16")
    debug(pages, copied_positions, k_copied, v_copied, 0)
    np.testing.assert_array_equal(k_copied.numpy()[0], k_np[:page_size])
    np.testing.assert_array_equal(v_copied.numpy()[0], v_np[:page_size])

    copy_length_indptr = tvm.runtime.tensor(np.array([0, 2], dtype="int32"))
    copy_src_dst_pos = tvm.runtime.tensor(np.array([[1, 5], [8, 9]], dtype="int32"))
    compact(pages, copy_length_indptr, copy_src_dst_pos, 1)
    compacted_positions = tvm.runtime.tensor(np.array([8, 9], dtype="int32"))
    k_compacted = tvm.runtime.empty((num_layers, 2, num_heads, head_dim), "float16")
    v_compacted = tvm.runtime.empty((num_layers, 2, num_heads, head_dim), "float16")
    debug(pages, compacted_positions, k_compacted, v_compacted, 0)
    np.testing.assert_array_equal(k_compacted.numpy()[0], k_np[[1, 5]])
    np.testing.assert_array_equal(v_compacted.numpy()[0], v_np[[1, 5]])


def test_vllm_cache_layout_runtime_append_debug_roundtrip():
    num_layers = 1
    num_qo_heads = 4
    num_kv_heads = 2
    head_dim = 8
    page_size = 4
    seq_len = 5
    dtype = "float16"
    device = tvm.cpu()
    target = "llvm"

    append = tvm.tirx.build(
        tvm_kv_cache._kv_cache_transpose_append_vllm_layout(
            num_kv_heads, head_dim, dtype, page_size
        ),
        target=target,
    ).main
    debug = tvm.tirx.build(
        tvm_kv_cache._kv_cache_debug_get_kv_vllm_layout(
            num_layers, num_kv_heads, head_dim, dtype, page_size
        ),
        target=target,
    ).main
    copy_page = tvm.tirx.build(
        tvm_kv_cache._copy_single_page_cpu_vllm_layout(
            num_kv_heads, page_size, head_dim, dtype
        ),
        target=target,
    ).main
    compact = tvm.tirx.build(
        tvm_kv_cache._compact_kv_copy_cpu_vllm_layout(
            num_kv_heads, head_dim, dtype, page_size
        ),
        target=target,
    ).main
    merge = tvm.tirx.build(tvm_kv_cache._merge_state_inplace_cpu(dtype), target=target).main
    split_rotary = tvm.tirx.build(
        tvm_kv_cache.llama_rope_with_position_map(
            10000.0, 1.0, head_dim, num_qo_heads, num_kv_heads, dtype, {}
        ),
        target=target,
    ).main

    create = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    cache = create(
        tvm.runtime.ShapeTuple([4, 64, 16, page_size, 0]),
        tvm.runtime.ShapeTuple([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        tvm.runtime.ShapeTuple([0]),
        False,
        0,
        1.0,
        10000.0,
        None,
        tvm.runtime.empty((), dtype, device=device),
        append,
        None,
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [merge],
        split_rotary,
        copy_page,
        debug,
        compact,
    )

    add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    append_mha_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_append_mha_kv")
    debug_get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")
    fork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")

    add_sequence(cache, 7)
    begin_forward(cache, tvm.runtime.ShapeTuple([7]), tvm.runtime.ShapeTuple([seq_len]), None)
    k_np = (
        np.arange(seq_len * num_kv_heads * head_dim).reshape(seq_len, num_kv_heads, head_dim) + 1
    ).astype(dtype)
    v_np = (k_np + 1000).astype(dtype)
    append_mha_kv(cache, 0, tvm.runtime.tensor(k_np, device), tvm.runtime.tensor(v_np, device))
    end_forward(cache)

    k_actual = tvm.runtime.empty((num_layers, seq_len, num_kv_heads, head_dim), dtype, device)
    v_actual = tvm.runtime.empty((num_layers, seq_len, num_kv_heads, head_dim), dtype, device)
    debug_get_kv(cache, 7, 0, seq_len, k_actual, v_actual)
    np.testing.assert_array_equal(k_actual.numpy()[0], k_np)
    np.testing.assert_array_equal(v_actual.numpy()[0], v_np)

    fork_sequence(cache, 7, 8, -1)
    begin_forward(cache, tvm.runtime.ShapeTuple([8]), tvm.runtime.ShapeTuple([1]), None)
    k_new = np.full((1, num_kv_heads, head_dim), 900, dtype=dtype)
    v_new = (k_new + 1000).astype(dtype)
    append_mha_kv(cache, 0, tvm.runtime.tensor(k_new, device), tvm.runtime.tensor(v_new, device))
    end_forward(cache)

    parent_k = tvm.runtime.empty((num_layers, seq_len, num_kv_heads, head_dim), dtype, device)
    parent_v = tvm.runtime.empty((num_layers, seq_len, num_kv_heads, head_dim), dtype, device)
    debug_get_kv(cache, 7, 0, seq_len, parent_k, parent_v)
    np.testing.assert_array_equal(parent_k.numpy()[0], k_np)
    np.testing.assert_array_equal(parent_v.numpy()[0], v_np)

    forked_k_expected = np.concatenate([k_np, k_new], axis=0)
    forked_v_expected = np.concatenate([v_np, v_new], axis=0)
    forked_k = tvm.runtime.empty(
        (num_layers, seq_len + 1, num_kv_heads, head_dim), dtype, device
    )
    forked_v = tvm.runtime.empty(
        (num_layers, seq_len + 1, num_kv_heads, head_dim), dtype, device
    )
    debug_get_kv(cache, 8, 0, seq_len + 1, forked_k, forked_v)
    np.testing.assert_array_equal(forked_k.numpy()[0], forked_k_expected)
    np.testing.assert_array_equal(forked_v.numpy()[0], forked_v_expected)


def test_nn_module_paged_decode_metadata_tensors():
    cache_var = relax.Var("cache", relax.ObjectStructInfo())
    bb = relax.BlockBuilder()
    with bb.function("forward", params=[cache_var]):
        cache = PagedKVCache(_expr=cache_var, _name="cache")
        _, page_indptr, _, _, _, _ = cache.get_paged_decode_metadata_tensors(
            0, kv_dtype="float16"
        )
        bb.emit_func_output(page_indptr)
    tvm_mod = bb.finalize()
    calls = []

    def visit(expr):
        if (
            isinstance(expr, relax.Call)
            and expr.op == tvm.ir.Op.get("relax.call_pure_packed")
            and isinstance(expr.args[0], relax.ExternFunc)
            and expr.args[0].global_symbol
            == "vm.builtin.attention_kv_cache_get_paged_decode_metadata"
        ):
            calls.append(expr)

    relax.analysis.post_order_visit(tvm_mod["forward"], visit)
    assert len(calls) == 1
    metadata_sinfo = calls[0].struct_info
    assert isinstance(metadata_sinfo, relax.TupleStructInfo)
    assert len(metadata_sinfo.fields) == 6
    expected = [
        (5, "float16"),
        (1, "int32"),
        (1, "int32"),
        (1, "int32"),
        (1, "int32"),
        (1, "int32"),
    ]
    for field, (ndim, dtype) in zip(metadata_sinfo.fields, expected):
        assert isinstance(field, relax.TensorStructInfo)
        assert field.ndim == ndim
        assert field.dtype == dtype


def test_nn_module_fa2_paged_decode_attention_call():
    cache_var = relax.Var("cache", relax.ObjectStructInfo())
    q_var = relax.Var("q", relax.TensorStructInfo((1, 1, 8, 128), "float16"))
    bb = relax.BlockBuilder()
    with bb.function("forward", params=[cache_var, q_var]):
        cache = PagedKVCache(_expr=cache_var, _name="cache")
        q = core.Tensor(_expr=q_var)
        out, _ = cache.fa2_paged_decode_attention(0, q, 128, 128**-0.5)
        bb.emit_func_output(out._expr)
    tvm_mod = bb.finalize()
    call_names = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm.ir.Op.get("relax.call_dps_packed"):
            assert isinstance(expr.args[0], relax.ExternFunc)
            call_names.append(expr.args[0].global_symbol)

    relax.analysis.post_order_visit(tvm_mod["forward"], visit)
    assert "vm.builtin.attention_kv_cache_fa2_paged_decode" in call_names
    assert tvm_mod["forward"].ret_struct_info == relax.TensorStructInfo(
        (1, 1, 8, 128), "float16"
    )


def test_nn_module_cross_attention_with_paged_metadata_call():
    cache_var = relax.Var("cache", relax.ObjectStructInfo())
    q_var = relax.Var("q", relax.TensorStructInfo((1, 1, 8, 128), "float16"))
    bb = relax.BlockBuilder()
    with bb.function("forward", params=[cache_var, q_var]):
        cache = PagedKVCache(_expr=cache_var, _name="cache")
        q = core.Tensor(_expr=q_var)
        out, _ = cache.cross_attention_with_paged_metadata(0, q, 128, 128**-0.5)
        bb.emit_func_output(out._expr)
    tvm_mod = bb.finalize()
    call_names = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm.ir.Op.get("relax.call_dps_packed"):
            assert isinstance(expr.args[0], relax.ExternFunc)
            call_names.append(expr.args[0].global_symbol)

    relax.analysis.post_order_visit(tvm_mod["forward"], visit)
    assert "vm.builtin.attention_kv_cache_cross_attention_with_paged_metadata" in call_names
    assert tvm_mod["forward"].ret_struct_info == relax.TensorStructInfo(
        (1, 1, 8, 128), "float16"
    )


def test_fa2_paged_decode_cuda_backend_matches_numpy_reference():
    device = tvm.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device is not available")
    backend = tvm.get_global_func("tvm.contrib.flash_attn.fa2_paged_decode", allow_missing=True)
    if backend is None:
        pytest.skip("FA2 paged decode CUDA backend is not registered")

    rng = np.random.default_rng(0)
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    page_size = 4
    num_pages = 2
    context_len = 5
    sm_scale = head_dim**-0.5

    q_np = rng.normal(size=(1, num_q_heads, head_dim)).astype("float16")
    k_np = rng.normal(size=(num_pages, page_size, num_kv_heads, head_dim)).astype("float16")
    v_np = rng.normal(size=(num_pages, page_size, num_kv_heads, head_dim)).astype("float16")
    block_table_np = np.array([[0, 1]], dtype="int32")
    seqused_k_np = np.array([context_len], dtype="int32")
    cu_seqlens_q_np = np.array([0, 1], dtype="int32")
    q_rope_position_np = np.array([0], dtype="int32")

    q = tvm.runtime.tensor(q_np, device)
    k_pages = tvm.runtime.tensor(k_np, device)
    v_pages = tvm.runtime.tensor(v_np, device)
    block_table = tvm.runtime.tensor(block_table_np, device)
    seqused_k = tvm.runtime.tensor(seqused_k_np, device)
    cu_seqlens_q = tvm.runtime.tensor(cu_seqlens_q_np, device)
    q_rope_position = tvm.runtime.tensor(q_rope_position_np, device)
    out = tvm.runtime.empty((1, num_q_heads, head_dim), "float16", device)
    lse = tvm.runtime.empty((1, num_q_heads), "float32", device)

    backend(
        q,
        k_pages,
        v_pages,
        block_table,
        seqused_k,
        cu_seqlens_q,
        q_rope_position,
        0,
        0,
        sm_scale,
        out,
        lse,
    )
    device.sync()

    out_ref = np.zeros((1, num_q_heads, head_dim), dtype="float32")
    lse_ref = np.zeros((1, num_q_heads), dtype="float32")
    kv_group = num_q_heads // num_kv_heads
    for q_head in range(num_q_heads):
        kv_head = q_head // kv_group
        scores = []
        values = []
        for token in range(context_len):
            logical_block = token // page_size
            page_offset = token - logical_block * page_size
            physical_page = block_table_np[0, logical_block]
            key = k_np[physical_page, page_offset, kv_head].astype("float32")
            value = v_np[physical_page, page_offset, kv_head].astype("float32")
            scores.append(np.dot(q_np[0, q_head].astype("float32"), key) * sm_scale)
            values.append(value)
        scores = np.asarray(scores, dtype="float32")
        max_score = np.max(scores)
        probs = np.exp(scores - max_score)
        probs /= np.sum(probs)
        out_ref[0, q_head] = np.sum(probs[:, None] * np.stack(values), axis=0)
        lse_ref[0, q_head] = np.log(np.sum(np.exp(scores - max_score))) + max_score

    np.testing.assert_allclose(out.numpy().astype("float32"), out_ref, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(lse.numpy(), lse_ref, rtol=1e-3, atol=1e-3)


def test_fa2_paged_decode_cuda_backend_vllm_layout_matches_numpy_reference():
    device = tvm.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device is not available")
    backend = tvm.get_global_func("tvm.contrib.flash_attn.fa2_paged_decode", allow_missing=True)
    if backend is None:
        pytest.skip("FA2 paged decode CUDA backend is not registered")

    rng = np.random.default_rng(1)
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 256
    page_size = 16
    num_pages = 2
    context_len = 17
    sm_scale = head_dim**-0.5

    q_np = rng.normal(size=(1, num_q_heads, head_dim)).astype("float16")
    k_logical = rng.normal(size=(num_pages, page_size, num_kv_heads, head_dim)).astype("float16")
    v_logical = rng.normal(size=(num_pages, page_size, num_kv_heads, head_dim)).astype("float16")
    x = 16 // k_logical.dtype.itemsize
    k_np = (
        k_logical.reshape(num_pages, page_size, num_kv_heads, head_dim // x, x)
        .transpose(0, 2, 3, 1, 4)
        .copy()
    )
    v_np = v_logical.transpose(0, 2, 3, 1).copy()
    block_table_np = np.array([[0, 1]], dtype="int32")
    seqused_k_np = np.array([context_len], dtype="int32")

    q = tvm.runtime.tensor(q_np, device)
    k_pages = tvm.runtime.tensor(k_np, device)
    v_pages = tvm.runtime.tensor(v_np, device)
    block_table = tvm.runtime.tensor(block_table_np, device)
    seqused_k = tvm.runtime.tensor(seqused_k_np, device)
    cu_seqlens_q = tvm.runtime.tensor(np.array([0, 1], dtype="int32"), device)
    q_rope_position = tvm.runtime.tensor(np.array([context_len - 1], dtype="int32"), device)
    out = tvm.runtime.empty((1, num_q_heads, head_dim), "float16", device)
    lse = tvm.runtime.empty((1, num_q_heads), "float32", device)

    backend(
        q,
        k_pages,
        v_pages,
        block_table,
        seqused_k,
        cu_seqlens_q,
        q_rope_position,
        0,
        0,
        sm_scale,
        out,
        lse,
    )
    device.sync()

    out_ref = np.zeros((1, num_q_heads, head_dim), dtype="float32")
    lse_ref = np.zeros((1, num_q_heads), dtype="float32")
    kv_group = num_q_heads // num_kv_heads
    for q_head in range(num_q_heads):
        kv_head = q_head // kv_group
        scores = []
        values = []
        for token in range(context_len):
            logical_block = token // page_size
            page_offset = token - logical_block * page_size
            physical_page = block_table_np[0, logical_block]
            key = k_logical[physical_page, page_offset, kv_head].astype("float32")
            value = v_logical[physical_page, page_offset, kv_head].astype("float32")
            scores.append(np.dot(q_np[0, q_head].astype("float32"), key) * sm_scale)
            values.append(value)
        scores = np.asarray(scores, dtype="float32")
        max_score = np.max(scores)
        probs = np.exp(scores - max_score)
        probs /= np.sum(probs)
        out_ref[0, q_head] = np.sum(probs[:, None] * np.stack(values), axis=0)
        lse_ref[0, q_head] = np.log(np.sum(np.exp(scores - max_score))) + max_score

    np.testing.assert_allclose(out.numpy().astype("float32"), out_ref, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(lse.numpy(), lse_ref, rtol=1e-3, atol=1e-3)


def test_fa2_paged_decode_runtime_dispatches_registered_backend():
    calls = []

    @tvm.register_global_func("tvm.contrib.flash_attn.fa2_paged_decode", override=True)
    def fake_fa2_paged_decode_backend(
        q_data,
        k_pages,
        v_pages,
        block_table,
        seqused_k,
        cu_seqlens_q,
        q_rope_position,
        layer_id,
        depth,
        sm_scale,
        o_data,
        lse_data,
    ):
        calls.append(
            {
                "q_shape": tuple(q_data.shape),
                "k_shape": tuple(k_pages.shape),
                "v_shape": tuple(v_pages.shape),
                "block_table_shape": tuple(block_table.shape),
                "seqused_k_shape": tuple(seqused_k.shape),
                "cu_seqlens_q_shape": tuple(cu_seqlens_q.shape),
                "q_rope_position_shape": tuple(q_rope_position.shape),
                "layer_id": layer_id,
                "depth": depth,
                "sm_scale": sm_scale,
                "o_shape": tuple(o_data.shape),
                "lse_shape": tuple(lse_data.shape),
            }
        )

    dispatch = tvm.get_global_func("vm.builtin.attention_kv_cache_fa2_paged_decode")
    device = tvm.cpu()
    q_data = tvm.runtime.empty((1, 8, 128), "float16", device)
    k_pages = tvm.runtime.empty((1, 16, 2, 128), "float16", device)
    v_pages = tvm.runtime.empty((1, 16, 2, 128), "float16", device)
    block_table = tvm.runtime.empty((1, 1), "int32", device)
    seqused_k = tvm.runtime.empty((1,), "int32", device)
    cu_seqlens_q = tvm.runtime.empty((2,), "int32", device)
    q_rope_position = tvm.runtime.empty((1,), "int32", device)
    o_data = tvm.runtime.empty((1, 8, 128), "float16", device)
    lse_data = tvm.runtime.empty((1, 8), "float32", device)

    dispatch(
        q_data,
        k_pages,
        v_pages,
        block_table,
        seqused_k,
        cu_seqlens_q,
        q_rope_position,
        3,
        0,
        128**-0.5,
        o_data,
        lse_data,
    )

    assert calls == [
        {
            "q_shape": (1, 8, 128),
            "k_shape": (1, 16, 2, 128),
            "v_shape": (1, 16, 2, 128),
            "block_table_shape": (1, 1),
            "seqused_k_shape": (1,),
            "cu_seqlens_q_shape": (2,),
            "q_rope_position_shape": (1,),
            "layer_id": 3,
            "depth": 0,
            "sm_scale": 128**-0.5,
            "o_shape": (1, 8, 128),
            "lse_shape": (1, 8),
        }
    ]


if __name__ == "__main__":
    test_nn_module_paged_kv_cache()
    test_nn_module_paged_decode_metadata_tensors()
    test_nn_module_fa2_paged_decode_metadata_tensors()
    monkeypatch = pytest.MonkeyPatch()
    test_nn_module_fa2_paged_decode_uses_vllm_cache_layout_sinfo(monkeypatch)
    monkeypatch.undo()
    monkeypatch = pytest.MonkeyPatch()
    test_vllm_cache_layout_selects_matching_page_helpers(monkeypatch)
    monkeypatch.undo()
    test_vllm_cache_layout_page_helpers_preserve_logical_kv()
    test_vllm_cache_layout_runtime_append_debug_roundtrip()
    test_nn_module_cross_attention_with_paged_metadata_call()
    test_nn_module_fa2_paged_decode_attention_call()
    test_fa2_paged_decode_cuda_backend_vllm_layout_matches_numpy_reference()
    test_fa2_paged_decode_runtime_dispatches_registered_backend()
