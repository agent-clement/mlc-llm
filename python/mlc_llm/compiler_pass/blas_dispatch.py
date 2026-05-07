"""A compiler pass that dispatches patterns to CUBLAS."""

import os

import tvm
from tvm import IRModule, relax
from tvm.relax.backend import get_patterns_with_prefix

try:
    import tvm.relax.backend.cuda.cublas as _cublas  # noqa: F401
    import tvm.relax.backend.rocm.hipblas as _hipblas  # noqa: F401
except ImportError:
    # Note: legacy path of cublas/hipblas for backward compatibility
    pass


@tvm.transform.module_pass(opt_level=0, name="BLASDispatch")
class BLASDispatch:
    """A compiler pass that dispatches patterns to cuBLAS/hipBLAS."""

    def __init__(self, target: tvm.target.Target) -> None:
        if target.kind.name == "cuda":
            self.has_blas = tvm.get_global_func("relax.ext.cublas", True)
            if not self.has_blas:
                raise Exception("cuBLAS is not enabled.")
            self.patterns = get_patterns_with_prefix("cublas")
        elif target.kind.name == "rocm":
            self.has_blas = tvm.get_global_func("relax.ext.hipblas", True)
            if not self.has_blas:
                raise Exception("hipBLAS is not enabled.")
            self.patterns = get_patterns_with_prefix("hipblas")
        else:
            raise Exception(f"Unsupported target {target.kind.name} for BLAS dispatch.")

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        model_names = [
            gv.name_hint for gv, func in mod.functions.items() if isinstance(func, relax.Function)
        ]
        # exclude single batch decode
        if os.environ.get("MLC_CUBLAS_SINGLE_DECODE", "0") != "1":
            model_names = [name for name in model_names if "batch" in name or "decode" not in name]
        # cuBLAS JSON codegen expects graph inputs to be tensors. Functions with shape
        # parameters, for example dynamic-shape vision embedding, can produce fused
        # external functions with R.Shape arguments and fail serialization.
        model_names = [
            name
            for name in model_names
            if not any(
                isinstance(param.struct_info, relax.ShapeStructInfo)
                for param in mod[name].params
            )
        ]
        mod = relax.transform.FuseOpsByPattern(
            self.patterns,
            bind_constants=False,
            annotate_codegen=True,
            entry_functions=model_names,
        )(mod)
        mod = tvm.transform.Sequential(
            [
                relax.transform.RunCodegen({}, entry_functions=model_names),
            ]
        )(mod)
        return mod
