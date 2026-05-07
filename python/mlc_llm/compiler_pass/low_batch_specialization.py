"""A compiler pass that dispatch low-batch-gemm to gemv schedule."""

import os
import tvm
from tvm import tirx
from tvm.ir.module import IRModule
from tvm.s_tir import dlight as dl


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        return default
    return parsed or default


@tvm.transform.module_pass(opt_level=0, name="LowBatchGemvSpecialize")
class LowBatchGemvSpecialize:
    """A compiler pass that dispatch low-batch-gemm to gemv schedule."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        for g_var, func in mod.functions_items():
            if isinstance(func, tirx.PrimFunc):
                if func.attrs and func.attrs.get("tirx.is_scheduled", 0):
                    continue
                low_batch_range = _parse_int_list_env("MLC_LOW_BATCH_GEMV_RANGES", [2, 8])
                buckets = _parse_int_list_env("MLC_LOW_BATCH_GEMV_BUCKETS", [2, 4])
                if len(low_batch_range) != len(buckets):
                    continue
                low_batch_funcs = []
                try:
                    for bucket in buckets:
                        low_batch_mod = IRModule({})
                        low_batch_mod["main"] = func
                        low_batch_mod = dl.ApplyDefaultSchedule(
                            dl.gpu.LowBatchGEMV(bucket),
                        )(low_batch_mod)
                        low_batch_funcs.append(low_batch_mod["main"])
                except AssertionError:
                    continue
                if any(
                    tvm.ir.structural_equal(low_batch_func, func)
                    for low_batch_func in low_batch_funcs
                ):
                    continue
                buffers = func.buffer_map.values()
                shapes = [buffer.shape for buffer in buffers]
                symbolic_vars = set(
                    expr for shape in shapes for expr in shape if isinstance(expr, tirx.Var)
                )
                if len(symbolic_vars) != 1:
                    continue
                gemm_mod = IRModule({})
                gemm_mod["main"] = func
                gemm_mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                )(gemm_mod)
                gemm_func = gemm_mod["main"]
                sym_var = next(iter(symbolic_vars))
                body = gemm_func.body
                for i, range_limit in reversed(list(enumerate(low_batch_range))):
                    body = tirx.IfThenElse(
                        tirx.op.tvm_thread_invariant(sym_var <= range_limit),
                        low_batch_funcs[i].body,
                        body,
                    )
                body = tirx.SBlock([], [], [], "root", body)
                body = tirx.SBlockRealize([], True, body)
                new_func = func.with_body(body)
                new_func = new_func.with_attr("tirx.is_scheduled", 1)
                new_func = new_func.with_attr("tirx.HoistIfThenElseExprWithBlock", 1)
                mod.update_func(g_var, new_func)
        return mod
