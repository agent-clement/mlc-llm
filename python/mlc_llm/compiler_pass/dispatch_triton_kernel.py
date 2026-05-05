"""A pass that dispatch generic calls of triton kernels to specific kernel implementations."""

from typing import List  # noqa: UP035

import tvm
from tvm import IRModule, relax
from tvm.relax.expr_functor import PyExprMutator, mutator

from mlc_llm.op.triton import (
    get_tir_chunk_delta_h_packed,
    get_tir_chunk_fwd_o,
    get_tir_chunk_fwd_o_packed,
    get_tir_chunk_local_cumsum_scalar,
    get_tir_chunk_scaled_dot_kkt,
    get_tir_lm_head_argmax_partial,
    get_tir_recompute_w_u,
    get_tir_solve_tril_64,
    get_tir_w8a8_block_fp8_group_matmul,
    get_tir_w8a8_block_fp8_matmul,
)
from mlc_llm.support import logging

logger = logging.getLogger(__name__)


@mutator
class _Rewriter(PyExprMutator):
    def __init__(self, mod: IRModule, target: tvm.target.Target) -> None:
        super().__init__(mod)
        self.mod = mod
        self.target = target
        self.extern_mods: List[tvm.runtime.Module] = []  # noqa: UP006

    def transform(self) -> tvm.IRModule:
        """Entry point of the transformation"""
        for g_var, func in self.mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            new_func = self.visit_expr(func)
            # new_func = remove_all_unused(new_func)
            self.builder_.update_func(g_var, new_func)

        mod = self.builder_.finalize()
        mod_attrs = dict(mod.attrs) if mod.attrs else {}
        mod = mod.with_attr(
            "external_mods", list(mod_attrs.get("external_mods", [])) + self.extern_mods
        )
        return mod

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)

        if (
            call.op != tvm.ir.Op.get("relax.call_dps_packed")
            or not isinstance(call.args[0], relax.ExternFunc)
            or not str(call.args[0].global_symbol).startswith("mlc.triton.")
        ):
            return call

        global_symbol = str(call.args[0].global_symbol)
        assert isinstance(call.args[1], relax.Tuple)
        if global_symbol == "mlc.triton.w8a8_block_fp8_matmul":
            return self.w8a8_block_fp8_matmul(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.w8a8_block_fp8_group_matmul":
            return self.w8a8_block_fp8_group_matmul(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.chunk_local_cumsum_scalar":
            return self.chunk_local_cumsum_scalar(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.chunk_scaled_dot_kkt":
            return self.chunk_scaled_dot_kkt(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.solve_tril_64":
            return self.solve_tril_64(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.recompute_w_u":
            return self.recompute_w_u(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.chunk_fwd_o":
            return self.chunk_fwd_o(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.chunk_delta_h_packed":
            return self.chunk_delta_h_packed(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.chunk_fwd_o_packed":
            return self.chunk_fwd_o_packed(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.lm_head_argmax_partial":
            return self.lm_head_argmax_partial(call.args[1].fields, call.struct_info)
        raise ValueError(f"Unknown mlc.triton kernel identifier: {global_symbol}")

    def w8a8_block_fp8_matmul(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the w8a8_block_fp8_matmul triton kernel."""
        assert len(args) == 16
        x, weight, x_scale, weight_scale = args[:4]
        (
            N,
            K,
            block_n,
            block_k,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[4:14]]
        in_dtype, out_dtype = str(args[14].value), str(args[15].value)

        prim_func, func_name = get_tir_w8a8_block_fp8_matmul(
            N,
            K,
            block_n,
            block_k,
            in_dtype,
            out_dtype,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            self.extern_mods,
        )
        if prim_func is None:
            # The TIR function is already in the IRModule
            gv = self.builder_.get().get_global_var(func_name)
        else:
            # Add the TIR function to the IRModule
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [x, weight, x_scale, weight_scale], out_sinfo=out_sinfo)

    def w8a8_block_fp8_group_matmul(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the w8a8_block_fp8_group_matmul triton kernel."""
        assert len(args) == 19
        x, weight, x_scale, weight_scale, expert_ids, indptr = args[:6]
        (
            N,
            K,
            num_experts,
            block_n,
            block_k,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[6:17]]
        in_dtype, out_dtype = str(args[17].value), str(args[18].value)

        prim_func, func_name = get_tir_w8a8_block_fp8_group_matmul(
            N,
            K,
            num_experts,
            block_n,
            block_k,
            in_dtype,
            out_dtype,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            self.extern_mods,
        )
        if prim_func is None:
            # The TIR function is already in the IRModule
            gv = self.builder_.get().get_global_var(func_name)
        else:
            # Add the TIR function to the IRModule
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(
            gv,
            [x, weight, x_scale, weight_scale, expert_ids, indptr],
            out_sinfo=out_sinfo,
        )

    def chunk_local_cumsum_scalar(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the Qwen3.5 GDN scalar chunk-local cumsum triton kernel."""
        assert len(args) == 4
        g = args[0]
        head_count, chunk_size, num_warps = [arg.value.value for arg in args[1:4]]

        prim_func, func_name = get_tir_chunk_local_cumsum_scalar(
            head_count,
            chunk_size,
            num_warps,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [g], out_sinfo=out_sinfo)

    def chunk_scaled_dot_kkt(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the Qwen3.5 GDN chunk-scaled K K^T triton kernel."""
        assert len(args) == 11
        k, beta, g = args[:3]
        (
            head_count,
            key_head_count,
            key_dim,
            chunk_size,
            block_k,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[3:10]]
        dtype = str(args[10].value)

        prim_func, func_name = get_tir_chunk_scaled_dot_kkt(
            head_count,
            key_head_count,
            key_dim,
            chunk_size,
            block_k,
            num_warps,
            num_stages,
            dtype,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [k, beta, g], out_sinfo=out_sinfo)

    def solve_tril_64(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the Qwen3.5 GDN BT=64 triangular inverse triton kernel."""
        assert len(args) == 5
        a = args[0]
        head_count, chunk_size, num_warps, num_stages = [
            arg.value.value for arg in args[1:5]
        ]
        prim_func, func_name = get_tir_solve_tril_64(
            head_count,
            chunk_size,
            num_warps,
            num_stages,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [a], out_sinfo=out_sinfo)

    def recompute_w_u(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the Qwen3.5 GDN W/U recomputation triton kernel."""
        assert len(args) == 15
        k, v, beta, a, g = args[:5]
        (
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_k,
            block_v,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[5:14]]
        dtype = str(args[14].value)

        prim_func, func_name = get_tir_recompute_w_u(
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_k,
            block_v,
            num_warps,
            num_stages,
            dtype,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [k, v, beta, a, g], out_sinfo=out_sinfo)

    def chunk_fwd_o(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the Qwen3.5 GDN chunk output triton kernel."""
        assert len(args) == 15
        q, k, v, h, g = args[:5]
        (
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_k,
            block_v,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[5:14]]
        dtype = str(args[14].value)

        prim_func, func_name = get_tir_chunk_fwd_o(
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_k,
            block_v,
            num_warps,
            num_stages,
            dtype,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [q, k, v, h, g], out_sinfo=out_sinfo)

    def chunk_delta_h_packed(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit the Qwen3.5 GDN packed chunk recurrence triton kernel."""
        assert len(args) == 14
        k, w, u, g, h0 = args[:5]
        (
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_v,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[5:13]]
        dtype = str(args[13].value)

        prim_func, func_name = get_tir_chunk_delta_h_packed(
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_v,
            num_warps,
            num_stages,
            dtype,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [k, w, u, g, h0], out_sinfo=out_sinfo)

    def chunk_fwd_o_packed(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit Qwen3.5 GDN chunk output from packed recurrence."""
        assert len(args) == 14
        q, k, packed, g = args[:4]
        (
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_k,
            block_v,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[4:13]]
        dtype = str(args[13].value)

        prim_func, func_name = get_tir_chunk_fwd_o_packed(
            head_count,
            key_head_count,
            key_dim,
            value_dim,
            chunk_size,
            block_k,
            block_v,
            num_warps,
            num_stages,
            dtype,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [q, k, packed, g], out_sinfo=out_sinfo)

    def lm_head_argmax_partial(
        self,
        args: List[relax.Expr],  # noqa: UP006
        out_sinfo: relax.StructInfo,
    ) -> relax.Expr:
        """Emit fused LM-head matvec + per-block argmax partials."""
        assert len(args) == 7
        hidden, weight = args[:2]
        vocab_size, hidden_size, block_m, block_k = [arg.value.value for arg in args[2:6]]
        dtype = str(args[6].value)

        prim_func, func_name = get_tir_lm_head_argmax_partial(
            vocab_size,
            hidden_size,
            block_m,
            block_k,
            dtype,
            self.extern_mods,
        )
        if prim_func is None:
            gv = self.builder_.get().get_global_var(func_name)
        else:
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [hidden, weight], out_sinfo=out_sinfo)


@tvm.transform.module_pass(opt_level=0, name="DispatchTritonKernel")
class DispatchTritonKernel:
    """Rewrite KV cache creation functions to IRModule."""

    def __init__(self, target: tvm.target.Target) -> None:
        """Initializer.

        Parameters
        ----------
        """
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        if self.target.kind.name != "cuda":
            return mod

        return _Rewriter(mod, self.target).transform()
