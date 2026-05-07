"""Operators enabled by external modules."""

from typing import List, Literal, Tuple  # noqa: UP035

import tvm
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import tirx as T

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# We use a wrapper function to avoid type annotation issue of "tl.constexpr" when
# triton is not installed.
def _get_triton_w8a8_block_fp8_gemm():
    # Triton kernel adapted from SGLang project
    # https://github.com/sgl-project/sglang/blob/v0.4.4/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py# noqa: E501
    def _triton_w8a8_block_fp8_gemm(
        # Pointers to inputs and output
        A,
        B,
        C,
        As,
        Bs,
        # Shape for matmul
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        # Stride for inputs and output
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_As_m: tl.constexpr,
        stride_As_k: tl.constexpr,
        stride_Bs_k: tl.constexpr,
        stride_Bs_n: tl.constexpr,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Triton-accelerated function used to perform linear operations (dot
        product) on input tensors `A` and `B` with block-wise quantization,
        and store the result in output tensor `C`.
        """

        pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        As_ptrs = As + offs_am * stride_As_m
        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn * stride_Bs_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
            b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    return _triton_w8a8_block_fp8_gemm


# We use a wrapper function to avoid type annotation issue of "tl.constexpr" when
# triton is not installed.
def _get_triton_w8a8_block_fp8_group_gemm():
    # Triton kernel adapted from SGLang project
    # https://github.com/sgl-project/sglang/blob/v0.4.4/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py# noqa: E501
    def _triton_w8a8_block_fp8_group_gemm(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        expert_ids_ptr,
        indptr_ptr,
        # Matrix dimensions
        EM,
        N: tl.constexpr,
        K: tl.constexpr,
        num_experts: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_be: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_asm: tl.constexpr,
        stride_ask: tl.constexpr,
        stride_bse: tl.constexpr,
        stride_bsk: tl.constexpr,
        stride_bsn: tl.constexpr,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        even_Ks: tl.constexpr,
    ):
        """
        Implements the fused computation for a Mixture of Experts (MOE) using
        token and expert matrices.

        Key Parameters:
        - A: The input tensor representing tokens with shape (*, K), where '*' can
            be any shape representing batches and K is the feature dimension of
            each token.
        - B: The stacked MOE weight tensor with shape (E, N, K), where E is
            the number of experts, K is the input feature dimension, and N is
            the output feature dimension.
        - C: The output cache tensor with shape (*, N), where '*' means the
            same shape as the input tensor A, and N is the output feature dimension.
        - expert_ids: A tensor containing the indices of the expert for each
            block. It determines which expert matrix from B should be used for
            each block in A.
        This kernel performs the multiplication of a token by its corresponding
        expert matrix as determined by `expert_ids`.
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M) + num_experts
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if expert_id == -1:
            return

        token_begin = tl.load(indptr_ptr + expert_id)
        token_end = tl.load(indptr_ptr + expert_id + 1)
        start_pid_m = tl.cdiv(token_begin, BLOCK_SIZE_M) + expert_id
        offs_token_id = (
            token_begin + (pid_m - start_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        )
        token_mask = offs_token_id < token_end

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_token_id[:, None] * stride_am + offs_k[None, :] * stride_ak

        b_ptrs = (
            b_ptr
            + expert_id * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
        offs_bsn = offs_bn // group_n
        b_scale_ptrs = b_scale_ptr + expert_id * stride_bse + offs_bsn * stride_bsn

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if even_Ks:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None],
                    other=0.0,
                )
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            # We accumulate along the K dimension.
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if c_ptr.dtype.element_ty == tl.bfloat16:
            accumulator = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float16:
            accumulator = accumulator.to(tl.float16)
        else:
            accumulator = accumulator.to(tl.float32)

        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    return _triton_w8a8_block_fp8_group_gemm


def _get_triton_lm_head_argmax_partial():
    # Fused vocabulary-block matvec + local argmax for greedy decode.
    # This avoids materializing full logits in the greedy-only fast path.
    def _triton_lm_head_argmax_partial(
        hidden_ptr,
        weight_ptr,
        partial_pairs_ptr,
        B,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_k = tl.arange(0, block_k)
        acc = tl.zeros((block_m,), dtype=tl.float32)

        for k0 in range(0, hidden_size, block_k):
            k = k0 + offs_k
            w = tl.load(
                weight_ptr + offs_m[:, None] * hidden_size + k[None, :],
                mask=(offs_m[:, None] < vocab_size) & (k[None, :] < hidden_size),
                other=0.0,
            )
            h = tl.load(
                hidden_ptr + pid_b * hidden_size + k,
                mask=(pid_b < B) & (k < hidden_size),
                other=0.0,
            )
            acc += tl.sum(w.to(tl.float32) * h[None, :].to(tl.float32), axis=1)

        acc = tl.where(offs_m < vocab_size, acc, -float("inf"))
        max_val = tl.max(acc, axis=0)
        max_idx = tl.min(tl.where(acc == max_val, offs_m, vocab_size), axis=0)
        num_blocks = tl.cdiv(vocab_size, block_m)
        partial_offset = (pid_b * num_blocks + pid_m) * 2
        tl.store(partial_pairs_ptr + partial_offset, max_val)
        tl.store(partial_pairs_ptr + partial_offset + 1, max_idx.to(tl.float32))

    return _triton_lm_head_argmax_partial


def _get_triton_chunk_local_cumsum_scalar():
    # Adapted from Flash Linear Attention's chunk_local_cumsum_scalar_kernel.
    # This is the equal-length, head-last, forward-only variant needed by the
    # Qwen3.5 GDN prefill path.
    def _triton_chunk_local_cumsum_scalar(
        s_ptr,
        o_ptr,
        T_len,
        H: tl.constexpr,
        BT: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        offsets_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offsets_t < T_len
        offsets = (pid_b * T_len + offsets_t) * H + pid_h
        vals = tl.load(s_ptr + offsets, mask=mask_t, other=0.0).to(tl.float32)
        out = tl.cumsum(vals, axis=0)
        tl.store(o_ptr + offsets, out, mask=mask_t)

    return _triton_chunk_local_cumsum_scalar


def _get_triton_chunk_scaled_dot_kkt():
    # Equal-length, head-last variant of FLA's chunk_scaled_dot_kkt_fwd_kernel.
    def _triton_chunk_scaled_dot_kkt(
        k_ptr,
        beta_ptr,
        g_ptr,
        a_ptr,
        T_len,
        H: tl.constexpr,
        Hg: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        pid_hg = pid_h // (H // Hg)

        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_bt = tl.arange(0, BT)
        mask_t = offs_t < T_len

        beta_offsets = (pid_b * T_len + offs_t) * H + pid_h
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=mask_t, other=0.0).to(tl.float32)

        acc = tl.zeros([BT, BT], dtype=tl.float32)
        for k_block in range(tl.cdiv(K, BK)):
            offs_k = k_block * BK + tl.arange(0, BK)
            mask_k = offs_k < K
            k_offsets = ((pid_b * T_len + offs_t[:, None]) * Hg + pid_hg) * K + offs_k[
                None, :
            ]
            k_vals = tl.load(k_ptr + k_offsets, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            k_beta = k_vals * beta_vals[:, None]
            acc += tl.dot(k_beta.to(k_vals.dtype), tl.trans(k_vals))

        g_offsets = (pid_b * T_len + offs_t) * H + pid_h
        g_vals = tl.load(g_ptr + g_offsets, mask=mask_t, other=0.0).to(tl.float32)
        acc = acc * tl.exp(g_vals[:, None] - g_vals[None, :])

        mask_a = (offs_t[:, None] > (pid_t * BT + offs_bt[None, :])) & (
            mask_t[:, None] & ((pid_t * BT + offs_bt[None, :]) < T_len)
        )
        acc = tl.where(mask_a, acc, 0.0)
        a_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * BT + offs_bt[None, :]
        tl.store(a_ptr + a_offsets, acc, mask=mask_t[:, None])

    return _triton_chunk_scaled_dot_kkt


def _get_triton_solve_tril_64():
    # Equal-length, non-TMA BT=64 variant of FLA's merge_16x16_to_64x64_inverse_kernel.
    def _triton_solve_tril_64(
        a_ptr,
        ai_ptr,
        T_len,
        H: tl.constexpr,
        BT: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        offs = tl.arange(0, 16)
        lower_mask = offs[:, None] > offs[None, :]
        eye_mask = offs[:, None] == offs[None, :]
        base_a = ((pid_b * T_len) * H + pid_h) * BT
        base_ai = base_a

        p_a_11 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT, 0),
            (16, 16),
            (1, 0),
        )
        p_a_22 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 16, 16),
            (16, 16),
            (1, 0),
        )
        p_a_33 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 32, 32),
            (16, 16),
            (1, 0),
        )
        p_a_44 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 48),
            (16, 16),
            (1, 0),
        )
        ai_11 = -tl.where(lower_mask, tl.load(p_a_11, boundary_check=(0, 1)).to(tl.float32), 0)
        ai_22 = -tl.where(lower_mask, tl.load(p_a_22, boundary_check=(0, 1)).to(tl.float32), 0)
        ai_33 = -tl.where(lower_mask, tl.load(p_a_33, boundary_check=(0, 1)).to(tl.float32), 0)
        ai_44 = -tl.where(lower_mask, tl.load(p_a_44, boundary_check=(0, 1)).to(tl.float32), 0)

        for i in range(2, min(16, T_len - pid_t * BT)):
            row = -tl.load(a_ptr + base_a + (pid_t * BT + i) * H * BT + offs)
            row += tl.sum(row[:, None] * ai_11, 0)
            ai_11 = tl.where((offs == i)[:, None], row, ai_11)
        for i in range(18, min(32, T_len - pid_t * BT)):
            row = -tl.load(a_ptr + base_a + (pid_t * BT + i) * H * BT + offs + 16)
            row += tl.sum(row[:, None] * ai_22, 0)
            ai_22 = tl.where((offs == i - 16)[:, None], row, ai_22)
        for i in range(34, min(48, T_len - pid_t * BT)):
            row = -tl.load(a_ptr + base_a + (pid_t * BT + i) * H * BT + offs + 32)
            row += tl.sum(row[:, None] * ai_33, 0)
            ai_33 = tl.where((offs == i - 32)[:, None], row, ai_33)
        for i in range(50, min(64, T_len - pid_t * BT)):
            row = -tl.load(a_ptr + base_a + (pid_t * BT + i) * H * BT + offs + 48)
            row += tl.sum(row[:, None] * ai_44, 0)
            ai_44 = tl.where((offs == i - 48)[:, None], row, ai_44)

        ai_11 += eye_mask
        ai_22 += eye_mask
        ai_33 += eye_mask
        ai_44 += eye_mask

        p_a_21 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 16, 0),
            (16, 16),
            (1, 0),
        )
        p_a_31 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 32, 0),
            (16, 16),
            (1, 0),
        )
        p_a_32 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 32, 16),
            (16, 16),
            (1, 0),
        )
        p_a_41 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 0),
            (16, 16),
            (1, 0),
        )
        p_a_42 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 16),
            (16, 16),
            (1, 0),
        )
        p_a_43 = tl.make_block_ptr(
            a_ptr + base_a,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 32),
            (16, 16),
            (1, 0),
        )
        a_21 = tl.load(p_a_21, boundary_check=(0, 1)).to(tl.float32)
        a_31 = tl.load(p_a_31, boundary_check=(0, 1)).to(tl.float32)
        a_32 = tl.load(p_a_32, boundary_check=(0, 1)).to(tl.float32)
        a_41 = tl.load(p_a_41, boundary_check=(0, 1)).to(tl.float32)
        a_42 = tl.load(p_a_42, boundary_check=(0, 1)).to(tl.float32)
        a_43 = tl.load(p_a_43, boundary_check=(0, 1)).to(tl.float32)

        ai_21 = -tl.dot(tl.dot(ai_22, a_21, input_precision="ieee"), ai_11, input_precision="ieee")
        ai_32 = -tl.dot(tl.dot(ai_33, a_32, input_precision="ieee"), ai_22, input_precision="ieee")
        ai_43 = -tl.dot(tl.dot(ai_44, a_43, input_precision="ieee"), ai_33, input_precision="ieee")
        ai_31 = -tl.dot(
            ai_33,
            tl.dot(a_31, ai_11, input_precision="ieee")
            + tl.dot(a_32, ai_21, input_precision="ieee"),
            input_precision="ieee",
        )
        ai_42 = -tl.dot(
            ai_44,
            tl.dot(a_42, ai_22, input_precision="ieee")
            + tl.dot(a_43, ai_32, input_precision="ieee"),
            input_precision="ieee",
        )
        ai_41 = -tl.dot(
            ai_44,
            tl.dot(a_41, ai_11, input_precision="ieee")
            + tl.dot(a_42, ai_21, input_precision="ieee")
            + tl.dot(a_43, ai_31, input_precision="ieee"),
            input_precision="ieee",
        )

        p_ai_11 = tl.make_block_ptr(
            ai_ptr + base_ai, (T_len, BT), (H * BT, 1), (pid_t * BT, 0), (16, 16), (1, 0)
        )
        p_ai_22 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 16, 16),
            (16, 16),
            (1, 0),
        )
        p_ai_33 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 32, 32),
            (16, 16),
            (1, 0),
        )
        p_ai_44 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 48),
            (16, 16),
            (1, 0),
        )
        p_ai_21 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 16, 0),
            (16, 16),
            (1, 0),
        )
        p_ai_31 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 32, 0),
            (16, 16),
            (1, 0),
        )
        p_ai_32 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 32, 16),
            (16, 16),
            (1, 0),
        )
        p_ai_41 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 0),
            (16, 16),
            (1, 0),
        )
        p_ai_42 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 16),
            (16, 16),
            (1, 0),
        )
        p_ai_43 = tl.make_block_ptr(
            ai_ptr + base_ai,
            (T_len, BT),
            (H * BT, 1),
            (pid_t * BT + 48, 32),
            (16, 16),
            (1, 0),
        )
        tl.store(p_ai_11, ai_11, boundary_check=(0, 1))
        tl.store(p_ai_22, ai_22, boundary_check=(0, 1))
        tl.store(p_ai_33, ai_33, boundary_check=(0, 1))
        tl.store(p_ai_44, ai_44, boundary_check=(0, 1))
        tl.store(p_ai_21, ai_21, boundary_check=(0, 1))
        tl.store(p_ai_31, ai_31, boundary_check=(0, 1))
        tl.store(p_ai_32, ai_32, boundary_check=(0, 1))
        tl.store(p_ai_41, ai_41, boundary_check=(0, 1))
        tl.store(p_ai_42, ai_42, boundary_check=(0, 1))
        tl.store(p_ai_43, ai_43, boundary_check=(0, 1))

    return _triton_solve_tril_64


def _get_triton_recompute_w_u():
    # Equal-length, head-last variant of FLA's recompute_w_u_fwd_kernel.
    def _triton_recompute_w_u(
        k_ptr,
        v_ptr,
        beta_ptr,
        a_ptr,
        g_ptr,
        wu_ptr,
        T_len,
        Hg: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        pid_hg = pid_h // (H // Hg)

        offs_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T_len
        beta_offsets = (pid_b * T_len + offs_t) * H + pid_h
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=mask_t, other=0.0)
        a_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * BT + tl.arange(
            0, BT
        )[None, :]
        a_vals = tl.load(a_ptr + a_offsets, mask=mask_t[:, None], other=0.0)

        for v_block in range(tl.cdiv(V, BV)):
            offs_v = v_block * BV + tl.arange(0, BV)
            mask_v = offs_v < V
            v_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * V + offs_v[
                None, :
            ]
            v_vals = tl.load(v_ptr + v_offsets, mask=mask_t[:, None] & mask_v[None, :], other=0.0)
            vb_vals = (v_vals * beta_vals[:, None]).to(v_vals.dtype)
            u_vals = tl.dot(a_vals.to(v_vals.dtype), vb_vals, allow_tf32=False)
            u_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * (K + V) + K + offs_v[
                None, :
            ]
            tl.store(wu_ptr + u_offsets, u_vals, mask=mask_t[:, None] & mask_v[None, :])

        g_offsets = (pid_b * T_len + offs_t) * H + pid_h
        g_exp = tl.exp(tl.load(g_ptr + g_offsets, mask=mask_t, other=0.0))
        for k_block in range(tl.cdiv(K, BK)):
            offs_k = k_block * BK + tl.arange(0, BK)
            mask_k = offs_k < K
            k_offsets = ((pid_b * T_len + offs_t[:, None]) * Hg + pid_hg) * K + offs_k[
                None, :
            ]
            k_vals = tl.load(k_ptr + k_offsets, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            kbg_vals = k_vals * (beta_vals * g_exp)[:, None]
            w_vals = tl.dot(a_vals.to(k_vals.dtype), kbg_vals.to(k_vals.dtype))
            wu_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * (K + V) + offs_k[
                None, :
            ]
            tl.store(wu_ptr + wu_offsets, w_vals, mask=mask_t[:, None] & mask_k[None, :])

    return _triton_recompute_w_u


def _get_triton_chunk_fwd_o():
    # Equal-length, head-last variant of FLA's chunk_fwd_kernel_o.
    def _triton_chunk_fwd_o(
        q_ptr,
        k_ptr,
        v_ptr,
        h_ptr,
        g_ptr,
        o_ptr,
        T_len,
        Hg: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_bh = tl.program_id(2)
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        pid_hg = pid_h // (H // Hg)
        nt = tl.cdiv(T_len, BT)

        offs_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T_len
        offs_k = tl.arange(0, BK)
        offs_v = pid_v * BV + tl.arange(0, BV)
        mask_v = offs_v < V

        out_vals = tl.zeros((BT, BV), dtype=tl.float32)
        attn_vals = tl.zeros((BT, BT), dtype=tl.float32)
        for k_block in range(tl.cdiv(K, BK)):
            cur_k = k_block * BK + offs_k
            mask_k = cur_k < K
            q_offsets = ((pid_b * T_len + offs_t[:, None]) * Hg + pid_hg) * K + cur_k[
                None, :
            ]
            k_offsets = ((pid_b * T_len + offs_t[None, :]) * Hg + pid_hg) * K + cur_k[
                :, None
            ]
            h_offsets = (((pid_b * nt + pid_t) * H + pid_h) * V + offs_v[:, None]) * K + cur_k[
                None, :
            ]
            q_vals = tl.load(q_ptr + q_offsets, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            k_vals = tl.load(k_ptr + k_offsets, mask=mask_k[:, None] & mask_t[None, :], other=0.0)
            h_vals = tl.load(h_ptr + h_offsets, mask=mask_v[:, None] & mask_k[None, :], other=0.0)
            out_vals += tl.dot(q_vals, tl.trans(h_vals), allow_tf32=False)
            attn_vals += tl.dot(q_vals, k_vals, allow_tf32=False)

        g_offsets = (pid_b * T_len + offs_t) * H + pid_h
        g_vals = tl.load(g_ptr + g_offsets, mask=mask_t, other=0.0)
        out_vals *= tl.exp(g_vals)[:, None]
        attn_vals *= tl.exp(g_vals[:, None] - g_vals[None, :])

        local_t = tl.arange(0, BT)
        attn_mask = (local_t[:, None] >= local_t[None, :]) & (mask_t[:, None] & mask_t[None, :])
        attn_vals = tl.where(attn_mask, attn_vals, 0.0)

        v_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * V + offs_v[None, :]
        v_vals = tl.load(v_ptr + v_offsets, mask=mask_t[:, None] & mask_v[None, :], other=0.0)
        scale = K ** -0.5
        out_vals = (out_vals + tl.dot(attn_vals.to(v_vals.dtype), v_vals)) * scale
        tl.store(o_ptr + v_offsets, out_vals, mask=mask_t[:, None] & mask_v[None, :])

    return _triton_chunk_fwd_o


def _get_triton_chunk_delta_h_packed():
    # Equal-length, head-last variant of FLA's chunk_gated_delta_rule_fwd_h.
    # The single output packs chunk states, v_new, and final state per [B, H].
    def _triton_chunk_delta_h_packed(
        k_ptr,
        w_ptr,
        u_ptr,
        g_ptr,
        h0_ptr,
        packed_ptr,
        T_len,
        Hg: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        pid_hg = pid_h // (H // Hg)
        nt = tl.cdiv(T_len, BT)
        total = nt * V * K + T_len * V + V * K
        packed_base = (pid_b * H + pid_h) * total

        offs_v = pid_v * BV + tl.arange(0, BV)
        mask_v = offs_v < V
        offs_t = tl.arange(0, BT)
        offs_k = tl.arange(0, 64)

        h0_base = ((pid_b * H + pid_h) * V + offs_v[:, None]) * K
        h1 = tl.load(
            h0_ptr + h0_base + offs_k[None, :],
            mask=mask_v[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        if K > 64:
            h2 = tl.load(
                h0_ptr + h0_base + 64 + offs_k[None, :],
                mask=mask_v[:, None] & ((64 + offs_k)[None, :] < K),
                other=0.0,
            ).to(tl.float32)

        for chunk_idx in range(nt):
            chunk_start = chunk_idx * BT
            cur_t = chunk_start + offs_t
            mask_t = cur_t < T_len

            h_store_base = packed_base + chunk_idx * V * K + offs_v[:, None] * K
            tl.store(
                packed_ptr + h_store_base + offs_k[None, :],
                h1,
                mask=mask_v[:, None] & (offs_k[None, :] < K),
            )
            if K > 64:
                tl.store(
                    packed_ptr + h_store_base + 64 + offs_k[None, :],
                    h2,
                    mask=mask_v[:, None] & ((64 + offs_k)[None, :] < K),
                )

            w_base = ((pid_b * T_len + cur_t[:, None]) * H + pid_h) * K
            u_base = ((pid_b * T_len + cur_t[:, None]) * H + pid_h) * V
            k_base = ((pid_b * T_len + cur_t[None, :]) * Hg + pid_hg) * K

            w1 = tl.load(
                w_ptr + w_base + offs_k[None, :],
                mask=mask_t[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )
            v_corr = tl.dot(w1, tl.trans(h1).to(w1.dtype), allow_tf32=False)
            if K > 64:
                w2 = tl.load(
                    w_ptr + w_base + 64 + offs_k[None, :],
                    mask=mask_t[:, None] & ((64 + offs_k)[None, :] < K),
                    other=0.0,
                )
                v_corr += tl.dot(w2, tl.trans(h2).to(w2.dtype), allow_tf32=False)

            u_vals = tl.load(
                u_ptr + u_base + offs_v[None, :],
                mask=mask_t[:, None] & mask_v[None, :],
                other=0.0,
            )
            v_new = u_vals.to(tl.float32) - v_corr

            vnew_store_base = packed_base + nt * V * K + cur_t[:, None] * V
            tl.store(
                packed_ptr + vnew_store_base + offs_v[None, :],
                v_new,
                mask=mask_t[:, None] & mask_v[None, :],
            )

            last_idx = tl.minimum((chunk_idx + 1) * BT, T_len) - 1
            g_last = tl.load(g_ptr + (pid_b * T_len + last_idx) * H + pid_h)
            g_vals = tl.load(
                g_ptr + (pid_b * T_len + cur_t) * H + pid_h,
                mask=mask_t,
                other=0.0,
            )
            gate_last = tl.exp(g_last)
            v_scaled = (v_new * tl.where(mask_t, tl.exp(g_last - g_vals), 0.0)[:, None]).to(
                u_vals.dtype
            )
            h1 *= gate_last
            if K > 64:
                h2 *= gate_last

            k1 = tl.load(
                k_ptr + k_base + offs_k[:, None],
                mask=(offs_k[:, None] < K) & mask_t[None, :],
                other=0.0,
            )
            h1 += tl.trans(tl.dot(k1, v_scaled, allow_tf32=False))
            if K > 64:
                k2 = tl.load(
                    k_ptr + k_base + 64 + offs_k[:, None],
                    mask=((64 + offs_k)[:, None] < K) & mask_t[None, :],
                    other=0.0,
                )
                h2 += tl.trans(tl.dot(k2, v_scaled, allow_tf32=False))

        final_base = packed_base + nt * V * K + T_len * V + offs_v[:, None] * K
        tl.store(
            packed_ptr + final_base + offs_k[None, :],
            h1,
            mask=mask_v[:, None] & (offs_k[None, :] < K),
        )
        if K > 64:
            tl.store(
                packed_ptr + final_base + 64 + offs_k[None, :],
                h2,
                mask=mask_v[:, None] & ((64 + offs_k)[None, :] < K),
            )

    return _triton_chunk_delta_h_packed


def _get_triton_chunk_fwd_o_packed():
    # Equal-length chunk output that consumes chunk_delta_h_packed output directly.
    def _triton_chunk_fwd_o_packed(
        q_ptr,
        k_ptr,
        packed_ptr,
        g_ptr,
        o_ptr,
        T_len,
        Hg: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_bh = tl.program_id(2)
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        pid_hg = pid_h // (H // Hg)
        nt = tl.cdiv(T_len, BT)
        total = nt * V * K + T_len * V + V * K
        packed_base = (pid_b * H + pid_h) * total

        offs_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T_len
        offs_k = tl.arange(0, BK)
        offs_v = pid_v * BV + tl.arange(0, BV)
        mask_v = offs_v < V

        out_vals = tl.zeros((BT, BV), dtype=tl.float32)
        attn_vals = tl.zeros((BT, BT), dtype=tl.float32)
        for k_block in range(tl.cdiv(K, BK)):
            cur_k = k_block * BK + offs_k
            mask_k = cur_k < K
            q_offsets = ((pid_b * T_len + offs_t[:, None]) * Hg + pid_hg) * K + cur_k[
                None, :
            ]
            k_offsets = ((pid_b * T_len + offs_t[None, :]) * Hg + pid_hg) * K + cur_k[
                :, None
            ]
            h_offsets = (
                packed_base
                + pid_t * V * K
                + offs_v[:, None] * K
                + cur_k[None, :]
            )
            q_vals = tl.load(q_ptr + q_offsets, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            k_vals = tl.load(k_ptr + k_offsets, mask=mask_k[:, None] & mask_t[None, :], other=0.0)
            h_vals = tl.load(
                packed_ptr + h_offsets,
                mask=mask_v[:, None] & mask_k[None, :],
                other=0.0,
            )
            out_vals += tl.dot(q_vals, tl.trans(h_vals).to(q_vals.dtype), allow_tf32=False)
            attn_vals += tl.dot(q_vals, k_vals, allow_tf32=False)

        g_offsets = (pid_b * T_len + offs_t) * H + pid_h
        g_vals = tl.load(g_ptr + g_offsets, mask=mask_t, other=0.0)
        out_vals *= tl.exp(g_vals)[:, None]
        attn_vals *= tl.exp(g_vals[:, None] - g_vals[None, :])

        local_t = tl.arange(0, BT)
        attn_mask = (local_t[:, None] >= local_t[None, :]) & (mask_t[:, None] & mask_t[None, :])
        attn_vals = tl.where(attn_mask, attn_vals, 0.0)

        v_offsets = packed_base + nt * V * K + offs_t[:, None] * V + offs_v[None, :]
        v_vals = tl.load(
            packed_ptr + v_offsets,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        out_vals = (out_vals + tl.dot(attn_vals, v_vals.to(attn_vals.dtype))) * (K ** -0.5)
        o_offsets = ((pid_b * T_len + offs_t[:, None]) * H + pid_h) * V + offs_v[None, :]
        tl.store(o_ptr + o_offsets, out_vals, mask=mask_t[:, None] & mask_v[None, :])

    return _triton_chunk_fwd_o_packed


def get_tir_w8a8_block_fp8_matmul(
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    in_dtype: Literal["float8_e4m3fn"],
    out_dtype: Literal["float16", "bfloat16"],
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    GROUP_SIZE_M: int,
    num_warps: int,
    num_stages: int,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for the w8a8_block_fp8_matmul kernel."""
    # NOTE: adding the type annotation of " -> Tuple[Optional[tvm.tirx.PrimFunc], str]"
    # will cause the failure of the type resolution in mypy.
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    name_suffix = f"_N{N}_K{K}_block_n{block_n}_block_k{block_k}_in{in_dtype}_out{out_dtype}"
    kernel_name = f"triton_w8a8_block_fp8_gemm{name_suffix}"
    tir_name = f"tir_w8a8_block_fp8_matmul{name_suffix}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_w8a8_block_fp8_gemm()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class BlockFP8Matmul:
        @T.prim_func(private=True)
        def tir_w8a8_block_fp8_matmul(
            var_A: T.handle,
            var_B: T.handle,
            var_As: T.handle,
            var_Bs: T.handle,
            var_C: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            M = T.SizeVar("M", "int32")
            A = T.match_buffer(var_A, (M, K), dtype=in_dtype)
            B = T.match_buffer(var_B, (N, K), dtype=in_dtype)
            As = T.match_buffer(var_As, (M, (K + block_k - 1) // block_k), "float32")
            Bs = T.match_buffer(
                var_Bs,
                ((N + block_n - 1) // block_n, (K + block_k - 1) // block_k),
                "float32",
            )
            C = T.match_buffer(var_C, (M, N), dtype=out_dtype)
            with T.sblock("root"):
                T.reads(
                    A[0:M, 0:K],
                    B[0:N, 0:K],
                    As[0:M, 0 : (K + block_k - 1) // block_k],
                    Bs[
                        0 : (N + block_n - 1) // block_n,
                        0 : (K + block_k - 1) // block_k,
                    ],
                )
                T.writes(C[0:M, 0:N])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(M, BLOCK_SIZE_M) * T.ceildiv(N, BLOCK_SIZE_N),),
                    A.data,
                    B.data,
                    C.data,
                    As.data,
                    Bs.data,
                    M,
                    N,
                    K,
                    K,  # stride_am
                    1,  # stride_ak
                    1,  # stride_bk
                    K,  # stride_bn
                    N,  # stride_cm
                    1,  # stride_cn
                    (K + block_k - 1) // block_k,  # stride_As_m
                    1,  # stride_As_k
                    1,  # stride_Bs_k
                    (K + block_k - 1) // block_k,  # stride_Bs_n
                    block_n,
                    block_k,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    GROUP_SIZE_M,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = BlockFP8Matmul.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return BlockFP8Matmul["tir_w8a8_block_fp8_matmul"], tir_name


def get_tir_w8a8_block_fp8_group_matmul(
    N: int,
    K: int,
    num_experts: int,
    block_n: int,
    block_k: int,
    in_dtype: Literal["float8_e4m3fn"],
    out_dtype: Literal["float16", "bfloat16"],
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    GROUP_SIZE_M: int,
    num_warps: int,
    num_stages: int,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for the w8a8_block_fp8_group_gemm kernel."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    name_suffix = (
        f"_N{N}_K{K}_num_experts{num_experts}_block_n{block_n}"
        f"_block_k{block_k}_in{in_dtype}_out{out_dtype}"
    )
    kernel_name = f"triton_w8a8_block_fp8_group_gemm{name_suffix}"
    tir_name = f"tir_w8a8_block_fp8_group_gemm{name_suffix}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_w8a8_block_fp8_group_gemm()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class BlockFP8GroupMatmul:
        @T.prim_func(private=True)
        def tir_w8a8_block_fp8_group_gemm(
            var_A: T.handle,
            var_B: T.handle,
            var_As: T.handle,
            var_Bs: T.handle,
            var_expert_ids: T.handle,
            var_indptr: T.handle,
            var_C: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            EM = T.SizeVar("EM", "int32")
            A = T.match_buffer(var_A, (EM, K), dtype=in_dtype)
            B = T.match_buffer(var_B, (num_experts, N, K), dtype=in_dtype)
            As = T.match_buffer(var_As, (EM, (K + block_k - 1) // block_k), "float32")
            Bs = T.match_buffer(
                var_Bs,
                (
                    num_experts,
                    (N + block_n - 1) // block_n,
                    (K + block_k - 1) // block_k,
                ),
                "float32",
            )
            expert_ids = T.match_buffer(
                var_expert_ids,
                ((EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,),
                "int32",
            )
            indptr = T.match_buffer(var_indptr, (num_experts + 1,), "int32")
            C = T.match_buffer(var_C, (EM, N), dtype=out_dtype)

            with T.sblock("root"):
                T.reads(
                    A[0:EM, 0:K],
                    B[0:num_experts, 0:N, 0:K],
                    As[0:EM, 0 : (K + block_k - 1) // block_k],
                    Bs[
                        0:num_experts,
                        0 : (N + block_n - 1) // block_n,
                        0 : (K + block_k - 1) // block_k,
                    ],
                    expert_ids[0 : (EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts],
                    indptr[0 : num_experts + 1],
                )
                T.writes(C[0:EM, 0:N])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    ((T.ceildiv(EM, BLOCK_SIZE_M) + num_experts) * T.ceildiv(N, BLOCK_SIZE_N),),
                    A.data,
                    B.data,
                    C.data,
                    As.data,
                    Bs.data,
                    expert_ids.data,
                    indptr.data,
                    EM,
                    N,
                    K,
                    num_experts,
                    K,  # stride_am
                    1,  # stride_ak
                    N * K,  # stride_be
                    1,  # stride_bk
                    K,  # stride_bn
                    N,  # stride_cm
                    1,  # stride_cn
                    (K + block_k - 1) // block_k,  # stride_asm
                    1,  # stride_ask
                    ((N + block_n - 1) // block_n) * ((K + block_k - 1) // block_k),  # stride_bse
                    1,  # stride_bsk
                    (K + block_k - 1) // block_k,  # stride_Bs_n
                    block_n,
                    block_k,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    GROUP_SIZE_M,
                    K % BLOCK_SIZE_K == 0,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = BlockFP8GroupMatmul.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return BlockFP8GroupMatmul["tir_w8a8_block_fp8_group_gemm"], tir_name


def get_tir_chunk_local_cumsum_scalar(
    H: int,
    chunk_size: int,
    num_warps: int,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for equal-length scalar chunk-local cumsum."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    kernel_name = f"triton_chunk_local_cumsum_scalar_H{H}_BT{chunk_size}"
    tir_name = f"tir_chunk_local_cumsum_scalar_H{H}_BT{chunk_size}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_chunk_local_cumsum_scalar()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class ChunkLocalCumsumScalar:
        @T.prim_func(private=True)
        def tir_chunk_local_cumsum_scalar(
            var_S: T.handle,
            var_O: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            S = T.match_buffer(var_S, (B, T_len, H), dtype="float32")
            O = T.match_buffer(var_O, (B, T_len, H), dtype="float32")
            with T.sblock("root"):
                T.reads(S[0:B, 0:T_len, 0:H])
                T.writes(O[0:B, 0:T_len, 0:H])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(T_len, chunk_size), B * H),
                    S.data,
                    O.data,
                    T_len,
                    H,
                    chunk_size,
                    num_warps=num_warps,
                )

    new_ext_mods = ChunkLocalCumsumScalar.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return ChunkLocalCumsumScalar["tir_chunk_local_cumsum_scalar"], tir_name


def get_tir_chunk_scaled_dot_kkt(
    H: int,
    Hg: int,
    K: int,
    chunk_size: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    dtype: str,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for equal-length chunk-scaled dot K K^T."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    kernel_name = f"triton_chunk_scaled_dot_kkt_H{H}_Hg{Hg}_K{K}_BT{chunk_size}_BK{block_k}_{dtype}"
    tir_name = f"tir_chunk_scaled_dot_kkt_H{H}_Hg{Hg}_K{K}_BT{chunk_size}_BK{block_k}_{dtype}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_chunk_scaled_dot_kkt()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class ChunkScaledDotKKT:
        @T.prim_func(private=True)
        def tir_chunk_scaled_dot_kkt(
            var_K: T.handle,
            var_Beta: T.handle,
            var_G: T.handle,
            var_A: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            K_buf = T.match_buffer(var_K, (B, T_len, Hg, K), dtype=dtype)
            Beta = T.match_buffer(var_Beta, (B, T_len, H), dtype="float32")
            G = T.match_buffer(var_G, (B, T_len, H), dtype="float32")
            A = T.match_buffer(var_A, (B, T_len, H, chunk_size), dtype="float32")
            with T.sblock("root"):
                T.reads(K_buf[0:B, 0:T_len, 0:Hg, 0:K], Beta[0:B, 0:T_len, 0:H], G[0:B, 0:T_len, 0:H])
                T.writes(A[0:B, 0:T_len, 0:H, 0:chunk_size])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(T_len, chunk_size), B * H),
                    K_buf.data,
                    Beta.data,
                    G.data,
                    A.data,
                    T_len,
                    H,
                    Hg,
                    K,
                    chunk_size,
                    block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = ChunkScaledDotKKT.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return ChunkScaledDotKKT["tir_chunk_scaled_dot_kkt"], tir_name


def get_tir_solve_tril_64(
    H: int,
    chunk_size: int,
    num_warps: int,
    num_stages: int,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for equal-length BT=64 triangular inverse."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")
    if chunk_size != 64:
        raise ValueError(f"solve_tril_64 only supports chunk_size=64, got {chunk_size}")

    kernel_name = f"triton_solve_tril_64_H{H}_BT{chunk_size}"
    tir_name = f"tir_solve_tril_64_H{H}_BT{chunk_size}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_solve_tril_64()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class SolveTril64:
        @T.prim_func(private=True)
        def tir_solve_tril_64(
            var_A: T.handle,
            var_Ai: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            A = T.match_buffer(var_A, (B, T_len, H, chunk_size), dtype="float32")
            Ai = T.match_buffer(var_Ai, (B, T_len, H, chunk_size), dtype="float32")
            with T.sblock("root"):
                T.reads(A[0:B, 0:T_len, 0:H, 0:chunk_size])
                T.writes(Ai[0:B, 0:T_len, 0:H, 0:chunk_size])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(T_len, chunk_size), B * H),
                    A.data,
                    Ai.data,
                    T_len,
                    H,
                    chunk_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = SolveTril64.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return SolveTril64["tir_solve_tril_64"], tir_name


def get_tir_recompute_w_u(
    H: int,
    Hg: int,
    K: int,
    V: int,
    chunk_size: int,
    block_k: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    dtype: str,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for equal-length GDN W/U recomputation."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    kernel_name = (
        f"triton_recompute_w_u_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BK{block_k}_BV{block_v}_{dtype}"
    )
    tir_name = (
        f"tir_recompute_w_u_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BK{block_k}_BV{block_v}_{dtype}"
    )
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_recompute_w_u()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class RecomputeWU:
        @T.prim_func(private=True)
        def tir_recompute_w_u(
            var_K: T.handle,
            var_V: T.handle,
            var_Beta: T.handle,
            var_A: T.handle,
            var_G: T.handle,
            var_WU: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            K_buf = T.match_buffer(var_K, (B, T_len, Hg, K), dtype=dtype)
            V_buf = T.match_buffer(var_V, (B, T_len, H, V), dtype=dtype)
            Beta = T.match_buffer(var_Beta, (B, T_len, H), dtype="float32")
            A = T.match_buffer(var_A, (B, T_len, H, chunk_size), dtype="float32")
            G = T.match_buffer(var_G, (B, T_len, H), dtype="float32")
            WU = T.match_buffer(var_WU, (B, T_len, H, K + V), dtype=dtype)
            with T.sblock("root"):
                T.reads(
                    K_buf[0:B, 0:T_len, 0:Hg, 0:K],
                    V_buf[0:B, 0:T_len, 0:H, 0:V],
                    Beta[0:B, 0:T_len, 0:H],
                    A[0:B, 0:T_len, 0:H, 0:chunk_size],
                    G[0:B, 0:T_len, 0:H],
                )
                T.writes(WU[0:B, 0:T_len, 0:H, 0 : K + V])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(T_len, chunk_size), B * H),
                    K_buf.data,
                    V_buf.data,
                    Beta.data,
                    A.data,
                    G.data,
                    WU.data,
                    T_len,
                    Hg,
                    H,
                    K,
                    V,
                    chunk_size,
                    block_k,
                    block_v,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = RecomputeWU.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return RecomputeWU["tir_recompute_w_u"], tir_name


def get_tir_chunk_fwd_o(
    H: int,
    Hg: int,
    K: int,
    V: int,
    chunk_size: int,
    block_k: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    dtype: str,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for equal-length GDN chunk output."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    kernel_name = (
        f"triton_chunk_fwd_o_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BK{block_k}_BV{block_v}_{dtype}"
    )
    tir_name = (
        f"tir_chunk_fwd_o_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BK{block_k}_BV{block_v}_{dtype}"
    )
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_chunk_fwd_o()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class ChunkFwdO:
        @T.prim_func(private=True)
        def tir_chunk_fwd_o(
            var_Q: T.handle,
            var_K: T.handle,
            var_V: T.handle,
            var_H: T.handle,
            var_G: T.handle,
            var_O: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            Q = T.match_buffer(var_Q, (B, T_len, Hg, K), dtype=dtype)
            K_buf = T.match_buffer(var_K, (B, T_len, Hg, K), dtype=dtype)
            V_buf = T.match_buffer(var_V, (B, T_len, H, V), dtype=dtype)
            H_buf = T.match_buffer(
                var_H,
                (B, (T_len + chunk_size - 1) // chunk_size, H, V, K),
                dtype=dtype,
            )
            G = T.match_buffer(var_G, (B, T_len, H), dtype="float32")
            O = T.match_buffer(var_O, (B, T_len, H, V), dtype=dtype)
            with T.sblock("root"):
                T.reads(
                    Q[0:B, 0:T_len, 0:Hg, 0:K],
                    K_buf[0:B, 0:T_len, 0:Hg, 0:K],
                    V_buf[0:B, 0:T_len, 0:H, 0:V],
                    H_buf[0:B, 0 : (T_len + chunk_size - 1) // chunk_size, 0:H, 0:V, 0:K],
                    G[0:B, 0:T_len, 0:H],
                )
                T.writes(O[0:B, 0:T_len, 0:H, 0:V])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(V, block_v), T.ceildiv(T_len, chunk_size), B * H),
                    Q.data,
                    K_buf.data,
                    V_buf.data,
                    H_buf.data,
                    G.data,
                    O.data,
                    T_len,
                    Hg,
                    H,
                    K,
                    V,
                    chunk_size,
                    block_k,
                    block_v,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = ChunkFwdO.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return ChunkFwdO["tir_chunk_fwd_o"], tir_name


def get_tir_chunk_delta_h_packed(
    H: int,
    Hg: int,
    K: int,
    V: int,
    chunk_size: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    dtype: str,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for equal-length packed GDN chunk recurrence."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")
    if K > 128:
        raise ValueError(f"chunk_delta_h_packed currently supports K <= 128, got {K}")

    kernel_name = (
        f"triton_chunk_delta_h_packed_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BV{block_v}_{dtype}"
    )
    tir_name = (
        f"tir_chunk_delta_h_packed_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BV{block_v}_{dtype}"
    )
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_chunk_delta_h_packed()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class ChunkDeltaHPacked:
        @T.prim_func(private=True)
        def tir_chunk_delta_h_packed(
            var_K: T.handle,
            var_W: T.handle,
            var_U: T.handle,
            var_G: T.handle,
            var_H0: T.handle,
            var_Packed: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            K_buf = T.match_buffer(var_K, (B, T_len, Hg, K), dtype=dtype)
            W = T.match_buffer(var_W, (B, T_len, H, K), dtype=dtype)
            U = T.match_buffer(var_U, (B, T_len, H, V), dtype=dtype)
            G = T.match_buffer(var_G, (B, T_len, H), dtype="float32")
            H0 = T.match_buffer(var_H0, (B, H, V, K), dtype="float32")
            Packed = T.match_buffer(
                var_Packed,
                (
                    B,
                    H,
                    ((T_len + chunk_size - 1) // chunk_size) * V * K + T_len * V + V * K,
                ),
                dtype="float32",
            )
            with T.sblock("root"):
                T.reads(
                    K_buf[0:B, 0:T_len, 0:Hg, 0:K],
                    W[0:B, 0:T_len, 0:H, 0:K],
                    U[0:B, 0:T_len, 0:H, 0:V],
                    G[0:B, 0:T_len, 0:H],
                    H0[0:B, 0:H, 0:V, 0:K],
                )
                T.writes(
                    Packed[
                        0:B,
                        0:H,
                        0 : ((T_len + chunk_size - 1) // chunk_size) * V * K
                        + T_len * V
                        + V * K,
                    ]
                )
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(V, block_v), B * H),
                    K_buf.data,
                    W.data,
                    U.data,
                    G.data,
                    H0.data,
                    Packed.data,
                    T_len,
                    Hg,
                    H,
                    K,
                    V,
                    chunk_size,
                    block_v,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = ChunkDeltaHPacked.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return ChunkDeltaHPacked["tir_chunk_delta_h_packed"], tir_name


def get_tir_lm_head_argmax_partial(
    vocab_size: int,
    hidden_size: int,
    block_m: int,
    block_k: int,
    dtype: str,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR wrapper for fused LM-head matvec + per-block argmax."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    kernel_name = (
        f"triton_lm_head_argmax_partial_V{vocab_size}_H{hidden_size}"
        f"_BM{block_m}_BK{block_k}_{dtype}"
    )
    tir_name = (
        f"tir_lm_head_argmax_partial_V{vocab_size}_H{hidden_size}"
        f"_BM{block_m}_BK{block_k}_{dtype}"
    )
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_lm_head_argmax_partial()
    triton_kernel.__name__ = kernel_name
    num_blocks = (vocab_size + block_m - 1) // block_m

    @I.ir_module
    class LMHeadArgmaxPartial:
        @T.prim_func(private=True)
        def tir_lm_head_argmax_partial(
            var_hidden: T.handle,
            var_weight: T.handle,
            var_partial_pairs: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            hidden = T.match_buffer(var_hidden, (B, hidden_size), dtype=dtype)
            weight = T.match_buffer(var_weight, (vocab_size, hidden_size), dtype=dtype)
            partial_pairs = T.match_buffer(var_partial_pairs, (B, num_blocks, 2), dtype="float32")
            with T.sblock("root"):
                T.reads(hidden[0:B, 0:hidden_size], weight[0:vocab_size, 0:hidden_size])
                T.writes(partial_pairs[0:B, 0:num_blocks, 0:2])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (num_blocks, B),
                    hidden.data,
                    weight.data,
                    partial_pairs.data,
                    B,
                    vocab_size,
                    hidden_size,
                    block_m,
                    block_k,
                    num_warps=8,
                    num_stages=4,
                )

    new_ext_mods = LMHeadArgmaxPartial.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return LMHeadArgmaxPartial["tir_lm_head_argmax_partial"], tir_name


def get_tir_chunk_fwd_o_packed(
    H: int,
    Hg: int,
    K: int,
    V: int,
    chunk_size: int,
    block_k: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    dtype: str,
    extern_mods: List[tvm.runtime.Module],  # noqa: UP006
):
    """Get the TIR function for chunk output from packed GDN recurrence."""
    if triton is None:
        raise RuntimeError("Triton is not installed. Please install it with `pip install triton`.")

    kernel_name = (
        f"triton_chunk_fwd_o_packed_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BK{block_k}_BV{block_v}_{dtype}"
    )
    tir_name = (
        f"tir_chunk_fwd_o_packed_H{H}_Hg{Hg}_K{K}_V{V}_BT{chunk_size}"
        f"_BK{block_k}_BV{block_v}_{dtype}"
    )
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]

    triton_kernel = _get_triton_chunk_fwd_o_packed()
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class ChunkFwdOPacked:
        @T.prim_func(private=True)
        def tir_chunk_fwd_o_packed(
            var_Q: T.handle,
            var_K: T.handle,
            var_Packed: T.handle,
            var_G: T.handle,
            var_O: T.handle,
        ):
            T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
            B = T.SizeVar("B", "int32")
            T_len = T.SizeVar("T_len", "int32")
            Q = T.match_buffer(var_Q, (B, T_len, Hg, K), dtype=dtype)
            K_buf = T.match_buffer(var_K, (B, T_len, Hg, K), dtype=dtype)
            Packed = T.match_buffer(
                var_Packed,
                (
                    B,
                    H,
                    ((T_len + chunk_size - 1) // chunk_size) * V * K + T_len * V + V * K,
                ),
                dtype="float32",
            )
            G = T.match_buffer(var_G, (B, T_len, H), dtype="float32")
            O = T.match_buffer(var_O, (B, T_len, H, V), dtype="float32")
            with T.sblock("root"):
                T.reads(
                    Q[0:B, 0:T_len, 0:Hg, 0:K],
                    K_buf[0:B, 0:T_len, 0:Hg, 0:K],
                    Packed[
                        0:B,
                        0:H,
                        0 : ((T_len + chunk_size - 1) // chunk_size) * V * K
                        + T_len * V
                        + V * K,
                    ],
                    G[0:B, 0:T_len, 0:H],
                )
                T.writes(O[0:B, 0:T_len, 0:H, 0:V])
                T.call_kernel(
                    triton.jit(triton_kernel),
                    (T.ceildiv(V, block_v), T.ceildiv(T_len, chunk_size), B * H),
                    Q.data,
                    K_buf.data,
                    Packed.data,
                    G.data,
                    O.data,
                    T_len,
                    Hg,
                    H,
                    K,
                    V,
                    chunk_size,
                    block_k,
                    block_v,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    new_ext_mods = ChunkFwdOPacked.attrs["external_mods"]
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return ChunkFwdOPacked["tir_chunk_fwd_o_packed"], tir_name


def _compute_expert_id_per_block(
    indptr: nn.Tensor,
    num_experts: int,
    M: nn.IntExpr,
    BLOCK_SIZE_M: int,
) -> nn.Tensor:
    """Compute the expert id for each threadblock (CTA).
    We assign an expert id to each threadblock, and the threadblock will
    compute the gemm with regard to the specified expert.

    Parameters
    ----------
    indptr : nn.Tensor
        The indptr tensor of group gemm, with shape of [num_experts + 1,].

    num_experts : int
        The number of total experts.

    M : nn.IntExpr
        The number of tokens.

    BLOCK_SIZE_M : int
        The block size of the threadblock along the batch dimension.

    Returns
    -------
    expert_ids : nn.Tensor
        The expert id for each threadblock, with shape of
        [(M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,].
    """

    @T.prim_func
    def tir_compute_expert_id_per_block(
        var_indptr: T.handle,
        var_expert_ids: T.handle,
        M: T.int64,
    ):
        T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
        indptr = T.match_buffer(var_indptr, (num_experts + 1,), "int32")
        expert_ids = T.match_buffer(
            var_expert_ids,
            ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,),
            "int32",
        )
        with T.sblock("root"):
            for eid in T.thread_binding(0, num_experts, thread="threadIdx.x"):
                start_block_id: T.int32 = (indptr[eid] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + eid
                num_blocks: T.int32 = (
                    indptr[eid + 1] - indptr[eid] + BLOCK_SIZE_M - 1
                ) // BLOCK_SIZE_M
                start_block_id_next_expert: T.int32 = (
                    (indptr[eid + 1] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + eid + 1
                )
                for block_id in T.serial(num_blocks):
                    expert_ids[start_block_id + block_id] = eid
                for block_id in T.serial(
                    start_block_id_next_expert - (start_block_id + num_blocks)
                ):
                    expert_ids[start_block_id + num_blocks + block_id] = -1

    assert num_experts <= 1024
    return nn.tensor_ir_op(
        tir_compute_expert_id_per_block,
        "tir_compute_expert_id_per_block",
        args=[indptr, M],
        out=nn.Tensor.placeholder(
            ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + num_experts,), dtype="int32"
        ),
    )


def chunk_local_cumsum_scalar(g: nn.Tensor, chunk_size: int = 64) -> nn.Tensor:
    """Triton chunk-local cumsum for Qwen3.5 GDN prefill gates.

    Input and output are shape ``[B, T, H]`` in float32. This intentionally
    covers only the equal-length, head-last, forward cumsum path.
    """
    if g.dtype != "float32":
        raise ValueError(f"chunk_local_cumsum_scalar expects float32 input, got {g.dtype}")
    if g.ndim != 3:
        raise ValueError(f"chunk_local_cumsum_scalar expects rank-3 input, got {g.ndim}")
    head_count = g.shape[2]
    if not isinstance(head_count, int):
        raise ValueError("chunk_local_cumsum_scalar requires static head count")
    return nn.extern(
        "mlc.triton.chunk_local_cumsum_scalar",
        args=[
            g,
            head_count,
            chunk_size,
            4,  # num_warps
        ],
        out=nn.Tensor.placeholder(g.shape, dtype="float32"),
    )


def chunk_scaled_dot_kkt(
    k: nn.Tensor,
    beta: nn.Tensor,
    g: nn.Tensor,
    chunk_size: int = 64,
    block_k: int = 64,
) -> nn.Tensor:
    """Triton beta-scaled, gate-scaled K K^T chunk matrix for Qwen3.5 GDN.

    Inputs are ``k: [B, T, Hg, K]`` and ``beta/g: [B, T, H]``. The output is
    ``[B, T, H, chunk_size]`` in float32, matching FLA's ``A`` layout.
    """
    if k.ndim != 4:
        raise ValueError(f"chunk_scaled_dot_kkt expects rank-4 k, got {k.ndim}")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("chunk_scaled_dot_kkt expects rank-3 beta and g")
    if beta.dtype != "float32" or g.dtype != "float32":
        raise ValueError(
            f"chunk_scaled_dot_kkt expects float32 beta/g, got beta={beta.dtype}, g={g.dtype}"
        )
    batch, seq_len, key_heads, key_dim = k.shape
    _, _, value_heads = beta.shape
    if not all(isinstance(x, int) for x in [key_heads, key_dim, value_heads]):
        raise ValueError("chunk_scaled_dot_kkt requires static head and key dimensions")
    if g.shape != beta.shape:
        raise ValueError(f"chunk_scaled_dot_kkt expects g shape {beta.shape}, got {g.shape}")
    return nn.extern(
        "mlc.triton.chunk_scaled_dot_kkt",
        args=[
            k,
            beta,
            g,
            int(value_heads),
            int(key_heads),
            int(key_dim),
            chunk_size,
            block_k,
            4,  # num_warps
            3,  # num_stages
            str(k.dtype),
        ],
        out=nn.Tensor.placeholder((batch, seq_len, value_heads, chunk_size), dtype="float32"),
    )


def solve_tril_64(a: nn.Tensor, chunk_size: int = 64) -> nn.Tensor:
    """Triton inverse of ``I + a`` for BT=64 GDN chunks.

    ``a`` is shape ``[B, T, H, 64]`` and should be strictly lower triangular
    inside each 64-token chunk.
    """
    if a.dtype != "float32":
        raise ValueError(f"solve_tril_64 expects float32 input, got {a.dtype}")
    if a.ndim != 4:
        raise ValueError(f"solve_tril_64 expects rank-4 input, got {a.ndim}")
    if chunk_size != 64:
        raise ValueError(f"solve_tril_64 only supports chunk_size=64, got {chunk_size}")
    head_count = a.shape[2]
    if not isinstance(head_count, int):
        raise ValueError("solve_tril_64 requires static head count")
    return nn.extern(
        "mlc.triton.solve_tril_64",
        args=[
            a,
            head_count,
            chunk_size,
            4,  # num_warps
            3,  # num_stages
        ],
        out=nn.Tensor.placeholder(a.shape, dtype="float32"),
    )


def recompute_w_u(
    k: nn.Tensor,
    v: nn.Tensor,
    beta: nn.Tensor,
    a: nn.Tensor,
    g: nn.Tensor,
    chunk_size: int = 64,
    block_k: int = 64,
    block_v: int = 64,
) -> nn.Tensor:
    """Triton recomputation of GDN ``w`` and ``u`` chunk matrices.

    Inputs are ``k: [B, T, Hg, K]``, ``v: [B, T, H, V]``,
    ``beta/g: [B, T, H]`` and ``a: [B, T, H, chunk_size]``. The output is
    ``[B, T, H, K + V]`` with ``w`` in ``[..., :K]`` and ``u`` in ``[..., K:]``.
    """
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("recompute_w_u expects rank-4 k and v")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("recompute_w_u expects rank-3 beta and g")
    if a.ndim != 4:
        raise ValueError(f"recompute_w_u expects rank-4 a, got {a.ndim}")
    if beta.dtype != "float32" or g.dtype != "float32" or a.dtype != "float32":
        raise ValueError(
            "recompute_w_u expects float32 beta/g/a, got "
            f"beta={beta.dtype}, g={g.dtype}, a={a.dtype}"
        )
    batch, seq_len, key_heads, key_dim = k.shape
    v_batch, v_seq_len, value_heads, value_dim = v.shape
    if (batch, seq_len) != (v_batch, v_seq_len):
        raise ValueError(f"recompute_w_u expects matching B/T, got k={k.shape}, v={v.shape}")
    beta_shape = [batch, seq_len, value_heads]
    if list(beta.shape) != beta_shape or list(g.shape) != list(beta.shape):
        raise ValueError(
            "recompute_w_u expects beta/g shape "
            f"{beta_shape}, got beta={beta.shape}, g={g.shape}"
        )
    a_shape = [batch, seq_len, value_heads, chunk_size]
    if list(a.shape) != a_shape:
        raise ValueError(
            "recompute_w_u expects a shape "
            f"{a_shape}, got {a.shape}"
        )
    if not all(isinstance(x, int) for x in [key_heads, key_dim, value_heads, value_dim]):
        raise ValueError("recompute_w_u requires static head and hidden dimensions")
    return nn.extern(
        "mlc.triton.recompute_w_u",
        args=[
            k,
            v,
            beta,
            a,
            g,
            int(value_heads),
            int(key_heads),
            int(key_dim),
            int(value_dim),
            chunk_size,
            block_k,
            block_v,
            4,  # num_warps
            3,  # num_stages
            str(k.dtype),
        ],
        out=nn.Tensor.placeholder((batch, seq_len, value_heads, key_dim + value_dim), dtype=k.dtype),
    )


def chunk_fwd_o(
    q: nn.Tensor,
    k: nn.Tensor,
    v: nn.Tensor,
    h: nn.Tensor,
    g: nn.Tensor,
    scale: float,
    chunk_size: int = 64,
    block_k: int = 64,
    block_v: int = 64,
) -> nn.Tensor:
    """Triton chunk output for Qwen3.5 GDN prefill.

    Inputs are ``q/k: [B, T, Hg, K]``, ``v: [B, T, H, V]``,
    ``h: [B, ceil(T / chunk_size), H, V, K]`` and ``g: [B, T, H]``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or h.ndim != 5:
        raise ValueError("chunk_fwd_o expects q/k/v rank-4 and h rank-5")
    if g.ndim != 3:
        raise ValueError(f"chunk_fwd_o expects rank-3 g, got {g.ndim}")
    if g.dtype != "float32":
        raise ValueError(f"chunk_fwd_o expects float32 g, got {g.dtype}")
    batch, seq_len, key_heads, key_dim = q.shape
    _, _, value_heads, value_dim = v.shape
    if list(k.shape) != list(q.shape):
        raise ValueError(f"chunk_fwd_o expects matching q/k shapes, got {q.shape} and {k.shape}")
    if list(g.shape) != [batch, seq_len, value_heads]:
        raise ValueError(
            "chunk_fwd_o expects g shape "
            f"{[batch, seq_len, value_heads]}, got {g.shape}"
        )
    if not all(isinstance(x, int) for x in [key_heads, key_dim, value_heads, value_dim]):
        raise ValueError("chunk_fwd_o requires static head and hidden dimensions")
    default_scale = int(key_dim) ** -0.5
    if abs(scale - default_scale) > 1e-7:
        raise ValueError(f"chunk_fwd_o currently supports scale={default_scale}, got {scale}")
    return nn.extern(
        "mlc.triton.chunk_fwd_o",
        args=[
            q,
            k,
            v,
            h,
            g,
            int(value_heads),
            int(key_heads),
            int(key_dim),
            int(value_dim),
            chunk_size,
            block_k,
            block_v,
            4,  # num_warps
            3,  # num_stages
            str(q.dtype),
        ],
        out=nn.Tensor.placeholder((batch, seq_len, value_heads, value_dim), dtype=v.dtype),
    )


def chunk_delta_h_packed(
    k: nn.Tensor,
    w: nn.Tensor,
    u: nn.Tensor,
    g: nn.Tensor,
    h0: nn.Tensor,
    chunk_size: int = 64,
    block_v: int = 64,
) -> nn.Tensor:
    """Triton packed chunk recurrence for Qwen3.5 GDN prefill.

    The output has shape ``[B, H, NT * V * K + T * V + V * K]``. Per head,
    it packs chunk states ``h``, then ``v_new``, then final state.
    """
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4:
        raise ValueError("chunk_delta_h_packed expects rank-4 k/w/u")
    if g.ndim != 3 or h0.ndim != 4:
        raise ValueError("chunk_delta_h_packed expects rank-3 g and rank-4 h0")
    if g.dtype != "float32" or h0.dtype != "float32":
        raise ValueError(
            f"chunk_delta_h_packed expects float32 g/h0, got g={g.dtype}, h0={h0.dtype}"
        )
    batch, seq_len, key_heads, key_dim = k.shape
    _, _, value_heads, value_dim = u.shape
    if list(w.shape) != [batch, seq_len, value_heads, key_dim]:
        raise ValueError(
            "chunk_delta_h_packed expects w shape "
            f"{[batch, seq_len, value_heads, key_dim]}, got {w.shape}"
        )
    if list(g.shape) != [batch, seq_len, value_heads]:
        raise ValueError(
            "chunk_delta_h_packed expects g shape "
            f"{[batch, seq_len, value_heads]}, got {g.shape}"
        )
    if list(h0.shape) != [batch, value_heads, value_dim, key_dim]:
        raise ValueError(
            "chunk_delta_h_packed expects h0 shape "
            f"{[batch, value_heads, value_dim, key_dim]}, got {h0.shape}"
        )
    if not all(isinstance(x, int) for x in [key_heads, key_dim, value_heads, value_dim]):
        raise ValueError("chunk_delta_h_packed requires static head and hidden dimensions")
    if int(key_dim) > 128:
        raise ValueError(f"chunk_delta_h_packed currently supports key_dim <= 128, got {key_dim}")
    chunks = (seq_len + chunk_size - 1) // chunk_size
    packed_dim = chunks * value_dim * key_dim + seq_len * value_dim + value_dim * key_dim
    return nn.extern(
        "mlc.triton.chunk_delta_h_packed",
        args=[
            k,
            w,
            u,
            g,
            h0,
            int(value_heads),
            int(key_heads),
            int(key_dim),
            int(value_dim),
            chunk_size,
            block_v,
            4,  # num_warps
            3,  # num_stages
            str(k.dtype),
        ],
        out=nn.Tensor.placeholder((batch, value_heads, packed_dim), dtype="float32"),
    )


def chunk_fwd_o_packed(
    q: nn.Tensor,
    k: nn.Tensor,
    packed: nn.Tensor,
    g: nn.Tensor,
    value_dim: int,
    chunk_size: int = 64,
    block_k: int = 64,
    block_v: int = 64,
) -> nn.Tensor:
    """Triton chunk output consuming ``chunk_delta_h_packed`` output directly."""
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("chunk_fwd_o_packed expects rank-4 q/k")
    if packed.ndim != 3 or g.ndim != 3:
        raise ValueError("chunk_fwd_o_packed expects rank-3 packed/g")
    if packed.dtype != "float32" or g.dtype != "float32":
        raise ValueError(
            f"chunk_fwd_o_packed expects float32 packed/g, got packed={packed.dtype}, g={g.dtype}"
        )
    batch, seq_len, key_heads, key_dim = q.shape
    _, value_heads, _ = packed.shape
    _, _, g_heads = g.shape
    if list(k.shape) != list(q.shape):
        raise ValueError(
            f"chunk_fwd_o_packed expects matching q/k shapes, got {q.shape} and {k.shape}"
        )
    if list(g.shape) != [batch, seq_len, value_heads] or g_heads != value_heads:
        raise ValueError(
            "chunk_fwd_o_packed expects g shape "
            f"{[batch, seq_len, value_heads]}, got {g.shape}"
        )
    if not all(isinstance(x, int) for x in [key_heads, key_dim, value_heads]):
        raise ValueError("chunk_fwd_o_packed requires static head and key dimensions")
    return nn.extern(
        "mlc.triton.chunk_fwd_o_packed",
        args=[
            q,
            k,
            packed,
            g,
            int(value_heads),
            int(key_heads),
            int(key_dim),
            int(value_dim),
            chunk_size,
            block_k,
            block_v,
            4,  # num_warps
            3,  # num_stages
            str(q.dtype),
        ],
        out=nn.Tensor.placeholder((batch, seq_len, value_heads, value_dim), dtype="float32"),
    )


def lm_head_argmax(
    hidden: nn.Tensor,
    weight: nn.Tensor,
    block_m: int = 32,
    block_k: int = 1024,
) -> nn.Tensor:
    """Fused LM-head matvec + top-1 for greedy decode."""
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("lm_head_argmax expects hidden [B, H] and weight [V, H]")
    if hidden.dtype != weight.dtype:
        raise ValueError(f"lm_head_argmax dtype mismatch: {hidden.dtype} vs {weight.dtype}")
    batch, hidden_size = hidden.shape
    vocab_size, weight_hidden = weight.shape
    if not isinstance(vocab_size, int) or not isinstance(hidden_size, int):
        raise ValueError("lm_head_argmax requires static vocab and hidden dimensions")
    if hidden_size != weight_hidden:
        raise ValueError(f"lm_head_argmax hidden mismatch: {hidden_size} vs {weight_hidden}")
    num_blocks = (vocab_size + block_m - 1) // block_m
    partial_pairs = nn.extern(
        "mlc.triton.lm_head_argmax_partial",
        args=[
            hidden,
            weight,
            vocab_size,
            hidden_size,
            block_m,
            block_k,
            str(hidden.dtype),
        ],
        out=nn.Tensor.placeholder((batch, num_blocks, 2), dtype="float32"),
    )

    @T.prim_func
    def tir_lm_head_argmax_final(
        var_partial_pairs: T.handle,
        var_token_ids: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        B = T.int64(is_size_var=True)
        partial_pairs_buf = T.match_buffer(var_partial_pairs, (B, num_blocks, 2), "float32")
        token_ids_buf = T.match_buffer(var_token_ids, (B,), "int32")
        shared_max = T.sblock_alloc_buffer((256,), dtype="float32", scope="shared")
        shared_idx = T.sblock_alloc_buffer((256,), dtype="int32", scope="shared")
        local_max = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")
        local_idx = T.sblock_alloc_buffer((1,), dtype="int32", scope="local")
        other_max = T.sblock_alloc_buffer((1,), dtype="float32", scope="local")
        other_idx = T.sblock_alloc_buffer((1,), dtype="int32", scope="local")
        for b_idx in T.thread_binding(B, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                with T.sblock("CTA"):
                    b = T.axis.spatial(B, b_idx)
                    t = T.axis.spatial(256, tx)
                    local_max[0] = T.min_value("float32")
                    local_idx[0] = T.int32(-1)
                    for i in T.serial(T.ceildiv(num_blocks, 256)):
                        block_idx = i * 256 + t
                        if block_idx < num_blocks:
                            value = partial_pairs_buf[b, block_idx, 0]
                            idx = T.cast(partial_pairs_buf[b, block_idx, 1], "int32")
                            if value > local_max[0]:
                                local_max[0] = value
                                local_idx[0] = idx
                            if value == local_max[0] and idx < local_idx[0]:
                                local_idx[0] = idx
                    shared_max[t] = local_max[0]
                    shared_idx[t] = local_idx[0]
                    T.tvm_storage_sync("shared")
                    for stride in T.serial(8):
                        offset = T.shift_right(256, stride + 1)
                        local_max[0] = shared_max[t]
                        local_idx[0] = shared_idx[t]
                        if t < offset:
                            other_max[0] = shared_max[t + offset]
                            other_idx[0] = shared_idx[t + offset]
                            if other_max[0] > local_max[0]:
                                local_max[0] = other_max[0]
                                local_idx[0] = other_idx[0]
                            if other_max[0] == local_max[0] and other_idx[0] < local_idx[0]:
                                local_idx[0] = other_idx[0]
                        shared_max[t] = local_max[0]
                        shared_idx[t] = local_idx[0]
                        T.tvm_storage_sync("shared")
                    if t == 0:
                        token_ids_buf[b] = local_idx[0]

    return nn.tensor_ir_op(
        tir_lm_head_argmax_final,
        "tir_lm_head_argmax_final",
        args=[partial_pairs],
        out=nn.Tensor.placeholder((batch,), dtype="int32"),
    )


def chunk_delta_h_final_state(
    packed: nn.Tensor,
    seq_len: nn.IntExpr,
    value_dim: int,
    key_dim: int,
    chunk_size: int = 64,
) -> nn.Tensor:
    """Extract ``[B, H, K, V]`` final state from ``chunk_delta_h_packed`` output."""
    if packed.ndim != 3:
        raise ValueError(f"chunk_delta_h_final_state expects rank-3 packed, got {packed.ndim}")
    if packed.dtype != "float32":
        raise ValueError(f"chunk_delta_h_final_state expects float32 packed, got {packed.dtype}")
    batch, heads, _ = packed.shape
    if not isinstance(heads, int):
        raise ValueError("chunk_delta_h_final_state requires static head count")

    @T.prim_func
    def tir_chunk_delta_h_final_state(
        var_Packed: T.handle,
        var_Out: T.handle,
        T_len: T.int64,
    ):
        T.func_attr({"op_pattern": 8, "tirx.is_scheduled": 1})
        B = T.SizeVar("B", "int32")
        D = T.SizeVar("D", "int32")
        Packed = T.match_buffer(var_Packed, (B, heads, D), dtype="float32")
        Out = T.match_buffer(var_Out, (B, heads, key_dim, value_dim), dtype="float32")
        final_base = T.ceildiv(T_len, chunk_size) * value_dim * key_dim + T_len * value_dim
        total = B * heads * key_dim * value_dim
        for bx in T.thread_binding(0, T.ceildiv(total, 256), thread="blockIdx.x"):
            for tx in T.thread_binding(0, 256, thread="threadIdx.x"):
                idx = bx * 256 + tx
                if idx < total:
                    v_idx = T.floormod(idx, value_dim)
                    tmp0 = idx // value_dim
                    k_idx = T.floormod(tmp0, key_dim)
                    tmp1 = tmp0 // key_dim
                    h_idx = T.floormod(tmp1, heads)
                    b_idx = tmp1 // heads
                    with T.sblock("extract_final_state"):
                        vb = T.axis.spatial(B, b_idx)
                        vh = T.axis.spatial(heads, h_idx)
                        vk = T.axis.spatial(key_dim, k_idx)
                        vv = T.axis.spatial(value_dim, v_idx)
                        T.reads(Packed[vb, vh, final_base + vv * key_dim + vk])
                        T.writes(Out[vb, vh, vk, vv])
                        Out[vb, vh, vk, vv] = Packed[vb, vh, final_base + vv * key_dim + vk]

    return nn.tensor_ir_op(
        tir_chunk_delta_h_final_state,
        "chunk_delta_h_final_state",
        args=[packed, seq_len],
        out=nn.Tensor.placeholder((batch, heads, key_dim, value_dim), dtype="float32"),
    )


def fp8_groupwise_scaled_gemm(
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    block_size: Tuple[int, int],  # noqa: UP006
    out_dtype: str,
) -> nn.Tensor:
    """Triton block-scale fp8 gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [m, k // block_size].

    weight : nn.Tensor
        The weight tensor, with shape of [n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [n // block_size, k // block_size].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim >= 2
    assert weight.ndim == 2
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim
    assert x.shape[-1] == weight.shape[1]
    assert x.shape[:-1] == x_scale.shape[:-1]
    assert (x.shape[-1] + block_size[1] - 1) // block_size[1] == x_scale.shape[-1]
    assert (weight.shape[1] + block_size[1] - 1) // block_size[1] == weight_scale.shape[1]
    assert (weight.shape[0] + block_size[0] - 1) // block_size[0] == weight_scale.shape[0]

    if x.dtype != "float8_e4m3fn" or weight.dtype != "float8_e4m3fn":
        raise ValueError(
            f"x and weight must be float8_e4m3fn, but got x={x.dtype}, weight={weight.dtype}"
        )
    if x_scale.dtype != "float32" and weight_scale.dtype != "float32":
        raise ValueError(
            "x_scale and weight_scale must be float32, but got "
            f"x_scale={x_scale.dtype}, weight_scale={weight_scale.dtype}"
        )
    if out_dtype not in ["float16", "bfloat16"]:
        raise ValueError(f"out_dtype must be float16 or bfloat16, but got {out_dtype}")

    M = x.shape[0]
    for i in range(1, x.ndim - 1):
        M *= x.shape[i]
    N = weight.shape[0]
    K = x.shape[-1]

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = block_size[0]
    BLOCK_SIZE_K = block_size[1]
    GROUP_SIZE_M = 32
    num_warps = 4
    num_stages = 3

    x_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(M, K)
    x_scale = x_scale.reshape(M, x_scale.shape[-1])

    out = nn.extern(
        "mlc.triton.w8a8_block_fp8_matmul",
        args=[
            x,
            weight,
            x_scale,
            weight_scale,
            N,
            K,
            block_size[0],
            block_size[1],
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            str(x.dtype),
            str(out_dtype),
        ],
        out=nn.Tensor.placeholder((M, N), dtype=out_dtype),
    )
    return out.reshape(*x_shape[:-1], N) if len(x_shape) > 2 else out


def fp8_groupwise_scaled_group_gemm(
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    indptr: nn.Tensor,
    block_size: Tuple[int, int],  # noqa: UP006
    out_dtype: str,
):
    """Triton block-scale fp8 group gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [m, k // block_size].

    weight : nn.Tensor
        The weight tensor, with shape of [num_experts, n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [num_experts, n // block_size, k // block_size].

    indptr : nn.Tensor
        The indptr tensor of group gemm, with shape of [num_experts + 1,].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim >= 2
    assert weight.ndim == 3
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim
    assert x.shape[-1] == weight.shape[2]
    assert (x.shape[-1] + block_size[1] - 1) // block_size[1] == x_scale.shape[-1]
    assert (weight.shape[2] + block_size[1] - 1) // block_size[1] == weight_scale.shape[2]
    assert (weight.shape[1] + block_size[0] - 1) // block_size[0] == weight_scale.shape[1]

    num_experts = weight.shape[0]
    M = x.shape[0]
    for i in range(1, x.ndim - 1):
        M *= x.shape[i]
    N = weight.shape[1]
    K = x.shape[-1]
    assert weight_scale.shape[0] == num_experts
    assert indptr.ndim == 1
    assert indptr.shape[0] == num_experts + 1

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = block_size[0]
    BLOCK_SIZE_K = block_size[1]
    GROUP_SIZE_M = 32
    num_warps = 4
    num_stages = 3

    x_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(M, K)
    x_scale = x_scale.reshape(M, x_scale.shape[-1])
    expert_ids = _compute_expert_id_per_block(indptr, num_experts, M, BLOCK_SIZE_M)

    out = nn.extern(
        "mlc.triton.w8a8_block_fp8_group_matmul",
        args=[
            x,
            weight,
            x_scale,
            weight_scale,
            expert_ids,
            indptr,
            N,
            K,
            num_experts,
            block_size[0],
            block_size[1],
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            str(x.dtype),
            str(out_dtype),
        ],
        out=nn.Tensor.placeholder((M, N), dtype=out_dtype),
    )
    return out.reshape(*x_shape[:-1], N) if len(x_shape) > 2 else out
