import numpy as np
import pytest
import torch
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op as nn_op
from tvm.relax.frontend.nn import spec

from mlc_llm.compiler_pass.dispatch_triton_kernel import DispatchTritonKernel
from mlc_llm.op import triton

pytestmark = [pytest.mark.op_correctness]


def _torch_chunk_local_cumsum(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        out[:, start:stop, :] = torch.cumsum(x[:, start:stop, :], dim=1)
    return out


def test_triton_chunk_local_cumsum_scalar():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 276
    head_count = 16
    chunk_size = 64

    class TestModule(nn.Module):
        def forward(self, x: nn.Tensor):
            return triton.chunk_local_cumsum_scalar(x, chunk_size=chunk_size)

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor(("batch", "seq", head_count), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    x_torch = torch.randn(batch_size, seq_len, head_count, dtype=torch.float32, device="cuda")
    expected = _torch_chunk_local_cumsum(x_torch, chunk_size)
    x_tvm = tvm.runtime.tensor(x_torch.cpu().numpy(), device=device)
    actual = vm["forward"](x_tvm).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=1e-5, rtol=1e-5)


def _torch_chunk_scaled_dot_kkt(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seq_len, key_heads, key_dim = k.shape
    value_heads = beta.shape[2]
    heads_per_group = value_heads // key_heads
    out = torch.zeros(batch, seq_len, value_heads, chunk_size, dtype=torch.float32, device=k.device)
    k_f32 = k.float()
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            kh = h_idx // heads_per_group
            for start in range(0, seq_len, chunk_size):
                stop = min(start + chunk_size, seq_len)
                for t_idx in range(start, stop):
                    for j_idx in range(start, t_idx):
                        local_j = j_idx - start
                        out[b_idx, t_idx, h_idx, local_j] = (
                            beta[b_idx, t_idx, h_idx]
                            * torch.dot(k_f32[b_idx, t_idx, kh], k_f32[b_idx, j_idx, kh])
                            * torch.exp(g[b_idx, t_idx, h_idx] - g[b_idx, j_idx, h_idx])
                        )
    return out


def test_triton_chunk_scaled_dot_kkt():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 96
    key_heads = 4
    value_heads = 4
    key_dim = 64
    chunk_size = 32

    class TestModule(nn.Module):
        def forward(self, k: nn.Tensor, beta: nn.Tensor, g: nn.Tensor):
            return triton.chunk_scaled_dot_kkt(k, beta, g, chunk_size=chunk_size, block_k=32)

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "k": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "beta": spec.Tensor(("batch", "seq", value_heads), "float32"),
                "g": spec.Tensor(("batch", "seq", value_heads), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    k_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    beta_torch = torch.sigmoid(
        torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
    )
    g_torch = _torch_chunk_local_cumsum(
        -torch.nn.functional.softplus(
            torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
        ),
        chunk_size,
    )
    expected = _torch_chunk_scaled_dot_kkt(k_torch, beta_torch, g_torch, chunk_size)
    actual = vm["forward"](
        tvm.runtime.tensor(k_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(beta_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(g_torch.cpu().numpy(), device=device),
    ).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=1e-2, rtol=1e-2)


def _torch_solve_tril_64(a: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch, seq_len, heads, _ = a.shape
    out = torch.zeros_like(a)
    for b_idx in range(batch):
        for h_idx in range(heads):
            for start in range(0, seq_len, chunk_size):
                stop = min(start + chunk_size, seq_len)
                size = stop - start
                mat = torch.eye(size, dtype=torch.float32, device=a.device)
                mat = mat + torch.tril(a[b_idx, start:stop, h_idx, :size], diagonal=-1)
                inv = torch.linalg.inv(mat)
                out[b_idx, start:stop, h_idx, :size] = inv
    return out


def test_triton_solve_tril_64():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 128
    heads = 4
    chunk_size = 64

    class TestModule(nn.Module):
        def forward(self, a: nn.Tensor):
            return triton.solve_tril_64(a, chunk_size=chunk_size)

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "a": spec.Tensor(("batch", "seq", heads, chunk_size), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    a_torch = torch.randn(
        batch_size, seq_len, heads, chunk_size, dtype=torch.float32, device="cuda"
    )
    local_col = torch.arange(chunk_size, device="cuda")
    local_row = torch.arange(seq_len, device="cuda") % chunk_size
    lower_mask = local_col.view(1, 1, 1, chunk_size) < local_row.view(1, seq_len, 1, 1)
    a_torch = torch.where(lower_mask, a_torch * 0.05, torch.zeros_like(a_torch))

    expected = _torch_solve_tril_64(a_torch, chunk_size)
    actual = vm["forward"](tvm.runtime.tensor(a_torch.cpu().numpy(), device=device)).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=1e-4, rtol=1e-4)


def _torch_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, key_heads, key_dim = k.shape
    _, _, value_heads, value_dim = v.shape
    heads_per_group = value_heads // key_heads
    w = torch.zeros(batch, seq_len, value_heads, key_dim, dtype=k.dtype, device=k.device)
    u = torch.zeros(batch, seq_len, value_heads, value_dim, dtype=v.dtype, device=v.device)
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            kh = h_idx // heads_per_group
            for start in range(0, seq_len, chunk_size):
                stop = min(start + chunk_size, seq_len)
                size = stop - start
                a_chunk = a[b_idx, start:stop, h_idx, :size].float()
                beta_chunk = beta[b_idx, start:stop, h_idx].float()
                g_chunk = torch.exp(g[b_idx, start:stop, h_idx].float())
                u_rhs = v[b_idx, start:stop, h_idx].float() * beta_chunk[:, None]
                w_rhs = k[b_idx, start:stop, kh].float() * (beta_chunk * g_chunk)[:, None]
                u[b_idx, start:stop, h_idx] = torch.matmul(a_chunk, u_rhs).to(v.dtype)
                w[b_idx, start:stop, h_idx] = torch.matmul(a_chunk, w_rhs).to(k.dtype)
    return w, u


def test_triton_recompute_w_u():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 128
    key_heads = 4
    value_heads = 4
    key_dim = 64
    value_dim = 128
    chunk_size = 64

    class TestModule(nn.Module):
        def forward(
            self,
            k: nn.Tensor,
            v: nn.Tensor,
            beta: nn.Tensor,
            a: nn.Tensor,
            g: nn.Tensor,
        ):
            return triton.recompute_w_u(
                k,
                v,
                beta,
                a,
                g,
                chunk_size=chunk_size,
                block_k=32,
                block_v=64,
            )

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "k": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "v": spec.Tensor(("batch", "seq", value_heads, value_dim), "float16"),
                "beta": spec.Tensor(("batch", "seq", value_heads), "float32"),
                "a": spec.Tensor(("batch", "seq", value_heads, chunk_size), "float32"),
                "g": spec.Tensor(("batch", "seq", value_heads), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    k_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    v_torch = torch.randn(
        batch_size, seq_len, value_heads, value_dim, dtype=torch.float16, device="cuda"
    )
    beta_torch = torch.sigmoid(
        torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
    )
    g_torch = _torch_chunk_local_cumsum(
        -torch.nn.functional.softplus(
            torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
        ),
        chunk_size,
    )
    a_torch = torch.randn(
        batch_size, seq_len, value_heads, chunk_size, dtype=torch.float32, device="cuda"
    )
    local_col = torch.arange(chunk_size, device="cuda")
    local_row = torch.arange(seq_len, device="cuda") % chunk_size
    lower_mask = local_col.view(1, 1, 1, chunk_size) <= local_row.view(1, seq_len, 1, 1)
    a_torch = torch.where(lower_mask, a_torch * 0.05, torch.zeros_like(a_torch))

    expected_w, expected_u = _torch_recompute_w_u(
        k_torch, v_torch, beta_torch, a_torch, g_torch, chunk_size
    )
    actual_wu = vm["forward"](
        tvm.runtime.tensor(k_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(v_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(beta_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(a_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(g_torch.cpu().numpy(), device=device),
    ).numpy()

    np.testing.assert_allclose(
        actual_wu[..., :key_dim], expected_w.cpu().numpy(), atol=1e-2, rtol=1e-2
    )
    np.testing.assert_allclose(
        actual_wu[..., key_dim:], expected_u.cpu().numpy(), atol=1e-2, rtol=1e-2
    )


def _torch_chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    batch, seq_len, key_heads, key_dim = q.shape
    _, _, value_heads, _ = v.shape
    heads_per_group = value_heads // key_heads
    out = torch.empty_like(v)
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            kh = h_idx // heads_per_group
            for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
                stop = min(start + chunk_size, seq_len)
                q_chunk = q[b_idx, start:stop, kh].float()
                k_chunk = k[b_idx, start:stop, kh].float()
                v_chunk = v[b_idx, start:stop, h_idx].float()
                h_state = h[b_idx, chunk_idx, h_idx].float()
                g_chunk = g[b_idx, start:stop, h_idx].float()
                local_o = torch.matmul(q_chunk, h_state.transpose(0, 1))
                local_o = local_o * torch.exp(g_chunk)[:, None]
                local_a = torch.matmul(q_chunk, k_chunk.transpose(0, 1))
                local_a = local_a * torch.exp(g_chunk[:, None] - g_chunk[None, :])
                local_a = torch.tril(local_a)
                out[b_idx, start:stop, h_idx] = (
                    (local_o + torch.matmul(local_a, v_chunk)) * scale
                ).to(v.dtype)
    return out


def test_triton_chunk_fwd_o():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 128
    key_heads = 2
    value_heads = 4
    key_dim = 64
    value_dim = 64
    chunk_size = 64
    scale = key_dim ** -0.5

    class TestModule(nn.Module):
        def forward(
            self,
            q: nn.Tensor,
            k: nn.Tensor,
            v: nn.Tensor,
            h: nn.Tensor,
            g: nn.Tensor,
        ):
            return triton.chunk_fwd_o(
                q,
                k,
                v,
                h,
                g,
                scale=scale,
                chunk_size=chunk_size,
                block_k=32,
                block_v=64,
            )

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "q": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "k": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "v": spec.Tensor(("batch", "seq", value_heads, value_dim), "float16"),
                "h": spec.Tensor(
                    ("batch", "chunks", value_heads, value_dim, key_dim), "float16"
                ),
                "g": spec.Tensor(("batch", "seq", value_heads), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    q_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    k_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    q_torch = torch.nn.functional.normalize(q_torch.float(), p=2, dim=-1).to(torch.float16)
    k_torch = torch.nn.functional.normalize(k_torch.float(), p=2, dim=-1).to(torch.float16)
    v_torch = torch.randn(
        batch_size, seq_len, value_heads, value_dim, dtype=torch.float16, device="cuda"
    )
    h_torch = (
        torch.randn(
            batch_size,
            (seq_len + chunk_size - 1) // chunk_size,
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float16,
            device="cuda",
        )
        * 0.05
    )
    g_torch = _torch_chunk_local_cumsum(
        -torch.nn.functional.softplus(
            torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
        ),
        chunk_size,
    )

    expected = _torch_chunk_fwd_o(
        q_torch, k_torch, v_torch, h_torch, g_torch, scale, chunk_size
    )
    actual = vm["forward"](
        tvm.runtime.tensor(q_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(k_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(v_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(h_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(g_torch.cpu().numpy(), device=device),
    ).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=2e-2, rtol=2e-2)


def _torch_chunk_delta_h_packed(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seq_len, key_heads, key_dim = k.shape
    _, _, value_heads, value_dim = u.shape
    heads_per_group = value_heads // key_heads
    chunks = (seq_len + chunk_size - 1) // chunk_size
    packed_dim = chunks * value_dim * key_dim + seq_len * value_dim + value_dim * key_dim
    packed = torch.zeros(batch, value_heads, packed_dim, dtype=torch.float32, device=k.device)
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            kh = h_idx // heads_per_group
            state = h0[b_idx, h_idx].clone().float()
            for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
                stop = min(start + chunk_size, seq_len)
                packed[
                    b_idx,
                    h_idx,
                    chunk_idx * value_dim * key_dim : (chunk_idx + 1) * value_dim * key_dim,
                ] = state.reshape(-1)
                k_chunk = k[b_idx, start:stop, kh].float()
                w_chunk = w[b_idx, start:stop, h_idx].float()
                u_chunk = u[b_idx, start:stop, h_idx].float()
                g_chunk = g[b_idx, start:stop, h_idx].float()
                v_new = u_chunk - torch.matmul(w_chunk, state.transpose(0, 1))
                vnew_base = chunks * value_dim * key_dim + start * value_dim
                packed[b_idx, h_idx, vnew_base : vnew_base + (stop - start) * value_dim] = (
                    v_new.reshape(-1)
                )
                g_last = g_chunk[-1]
                v_scaled = v_new * torch.exp(g_last - g_chunk)[:, None]
                state = state * torch.exp(g_last) + torch.matmul(v_scaled.transpose(0, 1), k_chunk)
            final_base = chunks * value_dim * key_dim + seq_len * value_dim
            packed[b_idx, h_idx, final_base:] = state.reshape(-1)
    return packed


def test_triton_chunk_delta_h_packed():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 128
    key_heads = 2
    value_heads = 4
    key_dim = 128
    value_dim = 64
    chunk_size = 64

    class TestModule(nn.Module):
        def forward(
            self,
            k: nn.Tensor,
            w: nn.Tensor,
            u: nn.Tensor,
            g: nn.Tensor,
            h0: nn.Tensor,
        ):
            return triton.chunk_delta_h_packed(
                k,
                w,
                u,
                g,
                h0,
                chunk_size=chunk_size,
                block_v=32,
            )

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "k": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "w": spec.Tensor(("batch", "seq", value_heads, key_dim), "float16"),
                "u": spec.Tensor(("batch", "seq", value_heads, value_dim), "float16"),
                "g": spec.Tensor(("batch", "seq", value_heads), "float32"),
                "h0": spec.Tensor(("batch", value_heads, value_dim, key_dim), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    k_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    w_torch = torch.randn(
        batch_size, seq_len, value_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    u_torch = torch.randn(
        batch_size, seq_len, value_heads, value_dim, dtype=torch.float16, device="cuda"
    )
    g_torch = _torch_chunk_local_cumsum(
        -torch.nn.functional.softplus(
            torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
        ),
        chunk_size,
    )
    h0_torch = torch.zeros(
        batch_size, value_heads, value_dim, key_dim, dtype=torch.float32, device="cuda"
    )

    expected = _torch_chunk_delta_h_packed(
        k_torch, w_torch, u_torch, g_torch, h0_torch, chunk_size
    )
    actual = vm["forward"](
        tvm.runtime.tensor(k_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(w_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(u_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(g_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(h0_torch.cpu().numpy(), device=device),
    ).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=3e-2, rtol=3e-2)


def _unpack_delta_h_packed(
    packed: torch.Tensor,
    seq_len: int,
    value_dim: int,
    key_dim: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, _ = packed.shape
    chunks = (seq_len + chunk_size - 1) // chunk_size
    h_size = chunks * value_dim * key_dim
    vnew_size = seq_len * value_dim
    h = packed[:, :, :h_size].reshape(batch, heads, chunks, value_dim, key_dim)
    h = h.permute(0, 2, 1, 3, 4).contiguous()
    v_new = packed[:, :, h_size : h_size + vnew_size].reshape(batch, heads, seq_len, value_dim)
    v_new = v_new.permute(0, 2, 1, 3).contiguous()
    final_state = packed[:, :, h_size + vnew_size :].reshape(batch, heads, value_dim, key_dim)
    return h, v_new, final_state


def test_triton_chunk_delta_h_final_state():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 276
    value_heads = 4
    key_dim = 128
    value_dim = 128
    chunk_size = 64
    chunks = (seq_len + chunk_size - 1) // chunk_size
    packed_dim = chunks * value_dim * key_dim + seq_len * value_dim + value_dim * key_dim

    class TestModule(nn.Module):
        def forward(self, packed: nn.Tensor, marker: nn.Tensor):
            return triton.chunk_delta_h_final_state(
                packed,
                marker.shape[1],
                value_dim=value_dim,
                key_dim=key_dim,
                chunk_size=chunk_size,
            )

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "packed": spec.Tensor(("batch", value_heads, packed_dim), "float32"),
                "marker": spec.Tensor(("batch", "seq", 1), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    packed_torch = torch.randn(
        batch_size, value_heads, packed_dim, dtype=torch.float32, device="cuda"
    )
    marker_torch = torch.empty(batch_size, seq_len, 1, dtype=torch.float32, device="cuda")
    _, _, final_state_vk = _unpack_delta_h_packed(
        packed_torch, seq_len, value_dim, key_dim, chunk_size
    )
    expected = final_state_vk.permute(0, 1, 3, 2).contiguous()
    actual = vm["forward"](
        tvm.runtime.tensor(packed_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(marker_torch.cpu().numpy(), device=device),
    ).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=1e-6, rtol=1e-6)


def test_triton_chunk_fwd_o_packed():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 128
    key_heads = 2
    value_heads = 4
    key_dim = 128
    value_dim = 64
    chunk_size = 64
    scale = key_dim ** -0.5

    class TestModule(nn.Module):
        def forward(self, q: nn.Tensor, k: nn.Tensor, packed: nn.Tensor, g: nn.Tensor):
            return triton.chunk_fwd_o_packed(
                q,
                k,
                packed,
                g,
                value_dim=value_dim,
                chunk_size=chunk_size,
                block_k=32,
                block_v=32,
            )

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "q": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "k": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "packed": spec.Tensor(("batch", value_heads, "packed_dim"), "float32"),
                "g": spec.Tensor(("batch", "seq", value_heads), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(0)
    q_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    k_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    w_torch = torch.randn(
        batch_size, seq_len, value_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    u_torch = torch.randn(
        batch_size, seq_len, value_heads, value_dim, dtype=torch.float16, device="cuda"
    )
    g_torch = _torch_chunk_local_cumsum(
        -torch.nn.functional.softplus(
            torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
        ),
        chunk_size,
    )
    h0_torch = torch.randn(
        batch_size, value_heads, value_dim, key_dim, dtype=torch.float32, device="cuda"
    ) * 0.01
    packed_torch = _torch_chunk_delta_h_packed(
        k_torch, w_torch, u_torch, g_torch, h0_torch, chunk_size
    )
    h_torch, v_new_torch, _ = _unpack_delta_h_packed(
        packed_torch, seq_len, value_dim, key_dim, chunk_size
    )
    expected = _torch_chunk_fwd_o(
        q_torch, k_torch, v_new_torch, h_torch, g_torch, scale, chunk_size
    )
    actual = vm["forward"](
        tvm.runtime.tensor(q_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(k_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(packed_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(g_torch.cpu().numpy(), device=device),
    ).numpy()

    np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=3e-2, rtol=3e-2)


def _torch_sequential_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, seq_len, key_heads, _ = k.shape
    _, _, value_heads, _ = v.shape
    heads_per_group = value_heads // key_heads
    out = torch.empty(batch, seq_len, value_heads, v.shape[-1], dtype=torch.float32, device=k.device)
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            kh = h_idx // heads_per_group
            state = h0[b_idx, h_idx].clone().float()
            for t_idx in range(seq_len):
                state = state * torch.exp(g[b_idx, t_idx, h_idx])
                dot_sk = torch.matmul(k[b_idx, t_idx, kh].float(), state.transpose(0, 1))
                delta = (v[b_idx, t_idx, h_idx].float() - dot_sk) * beta[b_idx, t_idx, h_idx]
                state = state + delta[:, None] * k[b_idx, t_idx, kh].float()[None, :]
                out[b_idx, t_idx, h_idx] = (
                    torch.matmul(q[b_idx, t_idx, kh].float(), state.transpose(0, 1)) * scale
                )
    return out


def _torch_sequential_gdn_with_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, key_heads, _ = k.shape
    _, _, value_heads, _ = v.shape
    heads_per_group = value_heads // key_heads
    out = torch.empty(batch, seq_len, value_heads, v.shape[-1], dtype=torch.float32, device=k.device)
    final_state = torch.empty_like(h0, dtype=torch.float32)
    for b_idx in range(batch):
        for h_idx in range(value_heads):
            kh = h_idx // heads_per_group
            state = h0[b_idx, h_idx].clone().float()
            for t_idx in range(seq_len):
                state = state * torch.exp(g[b_idx, t_idx, h_idx])
                dot_sk = torch.matmul(k[b_idx, t_idx, kh].float(), state.transpose(0, 1))
                delta = (v[b_idx, t_idx, h_idx].float() - dot_sk) * beta[b_idx, t_idx, h_idx]
                state = state + delta[:, None] * k[b_idx, t_idx, kh].float()[None, :]
                out[b_idx, t_idx, h_idx] = (
                    torch.matmul(q[b_idx, t_idx, kh].float(), state.transpose(0, 1)) * scale
                )
            final_state[b_idx, h_idx] = state
    return out, final_state


def test_triton_chunked_gdn_pipeline_real_shape_matches_sequential():
    if not tvm.cuda().exist:
        pytest.skip("CUDA is required")

    batch_size = 1
    seq_len = 276
    key_heads = 16
    value_heads = 16
    key_dim = 128
    value_dim = 128
    chunk_size = 64
    scale = key_dim ** -0.5

    class TestModule(nn.Module):
        def forward(
            self,
            q: nn.Tensor,
            k: nn.Tensor,
            v: nn.Tensor,
            beta: nn.Tensor,
            g: nn.Tensor,
            h0: nn.Tensor,
        ):
            g_cumsum = triton.chunk_local_cumsum_scalar(g, chunk_size=chunk_size)
            a = triton.chunk_scaled_dot_kkt(k, beta, g_cumsum, chunk_size=chunk_size, block_k=64)
            ai = triton.solve_tril_64(a, chunk_size=chunk_size)
            wu = triton.recompute_w_u(
                k,
                v,
                beta,
                ai,
                g_cumsum,
                chunk_size=chunk_size,
                block_k=64,
                block_v=64,
            )
            w, u = nn_op.split(wu, [key_dim], axis=3)
            packed = triton.chunk_delta_h_packed(
                k,
                w,
                u,
                g_cumsum,
                h0,
                chunk_size=chunk_size,
                block_v=64,
            )
            out = triton.chunk_fwd_o_packed(
                q,
                k,
                packed,
                g_cumsum,
                value_dim=value_dim,
                chunk_size=chunk_size,
                block_k=64,
                block_v=64,
            )
            final_state = triton.chunk_delta_h_final_state(
                packed,
                q.shape[1],
                value_dim=value_dim,
                key_dim=key_dim,
                chunk_size=chunk_size,
            )
            return out, final_state

    mod, _, _ = TestModule().export_tvm(
        spec={
            "forward": {
                "q": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "k": spec.Tensor(("batch", "seq", key_heads, key_dim), "float16"),
                "v": spec.Tensor(("batch", "seq", value_heads, value_dim), "float16"),
                "beta": spec.Tensor(("batch", "seq", value_heads), "float32"),
                "g": spec.Tensor(("batch", "seq", value_heads), "float32"),
                "h0": spec.Tensor(("batch", value_heads, value_dim, key_dim), "float32"),
            },
        },
        allow_extern=True,
    )
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    mod = DispatchTritonKernel(target)(mod)
    executable = relax.build(
        mod,
        target=target,
        relax_pipeline=relax.backend.cuda.get_default_pipeline(target),
    )
    vm = relax.VirtualMachine(executable, device)

    torch.manual_seed(1)
    q_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    k_torch = torch.randn(
        batch_size, seq_len, key_heads, key_dim, dtype=torch.float16, device="cuda"
    )
    q_torch = torch.nn.functional.normalize(q_torch.float(), p=2, dim=-1).to(torch.float16)
    k_torch = torch.nn.functional.normalize(k_torch.float(), p=2, dim=-1).to(torch.float16)
    v_torch = (
        torch.randn(batch_size, seq_len, value_heads, value_dim, dtype=torch.float16, device="cuda")
        * 0.1
    )
    beta_torch = torch.sigmoid(
        torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
    )
    g_torch = -torch.nn.functional.softplus(
        torch.randn(batch_size, seq_len, value_heads, dtype=torch.float32, device="cuda")
    )
    h0_torch = (
        torch.randn(batch_size, value_heads, value_dim, key_dim, dtype=torch.float32, device="cuda")
        * 0.01
    )

    expected_out, expected_state_vk = _torch_sequential_gdn_with_state(
        q_torch, k_torch, v_torch, beta_torch, g_torch, h0_torch, scale
    )
    expected_state = expected_state_vk.permute(0, 1, 3, 2).contiguous()
    actual_out, actual_state = vm["forward"](
        tvm.runtime.tensor(q_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(k_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(v_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(beta_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(g_torch.cpu().numpy(), device=device),
        tvm.runtime.tensor(h0_torch.cpu().numpy(), device=device),
    )

    np.testing.assert_allclose(actual_out.numpy(), expected_out.cpu().numpy(), atol=8e-2, rtol=8e-2)
    np.testing.assert_allclose(
        actual_state.numpy(), expected_state.cpu().numpy(), atol=8e-2, rtol=8e-2
    )
