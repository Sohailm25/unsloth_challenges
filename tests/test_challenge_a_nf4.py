# ABOUTME: Tests NF4 fused Triton dequantization behaviors for challenge A.
# ABOUTME: Covers correctness, torch.compile path, and optional asm/cache flags.

import pytest
import torch

from challenges.challenge_a_nf4 import (
    MLP,
    mlp_forward,
    mlp_dequantize,
    test_dequantize as run_test_dequantize,
    your_dequantize_nf4,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is required for NF4 tests",
)


@cuda_only
def test_your_dequantize_matches_reference():
    # Uses the provided harness which asserts correctness internally across fp16/bf16.
    elapsed = run_test_dequantize(your_dequantize_nf4)
    assert elapsed > 0


@cuda_only
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_torch_compile_path():
    torch.manual_seed(123)
    mlp = MLP(hd=64, m=128, dtype=torch.float16)
    x = torch.randn((1, 2, 64), device="cuda", dtype=torch.float16)

    class DequantMLP(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input):
            return mlp_forward(input, self.inner, your_dequantize_nf4)

    eager = DequantMLP(mlp).cuda()
    compiled = torch.compile(DequantMLP(mlp).cuda())

    out_eager = eager(x)
    out_compiled = compiled(x)

    assert torch.allclose(out_eager, out_compiled, atol=1e-1, rtol=1e-1)


@cuda_only
def test_optional_flags_do_not_change_shape():
    torch.manual_seed(321)
    mlp = MLP(hd=32, m=64, dtype=torch.float16)
    x = torch.randn((1, 1, 32), device="cuda", dtype=torch.float16)

    # Dequantize weights explicitly using optional flags to ensure paths are wired.
    up_w = your_dequantize_nf4(mlp.up_proj, use_custom_asm=True, use_cache_eviction=True)
    assert up_w.shape == mlp.up_proj.weight.quant_state.shape
    assert up_w.dtype == mlp.up_proj.weight.quant_state.dtype

    # Sanity check forward still runs with these flags via mlp_dequantize helper.
    a, b, c = mlp_dequantize(x, mlp, lambda w: your_dequantize_nf4(w, use_custom_asm=True, use_cache_eviction=True))
    assert a.shape[1] == mlp.up_proj.in_features
    assert b.shape[1] == mlp.gate_proj.in_features
    assert c.shape[1] == mlp.down_proj.in_features


@cuda_only
def test_out_buffer_reused_when_not_provided():
    torch.manual_seed(7)
    mlp = MLP(hd=32, m=64, dtype=torch.float16)
    # Two calls without supplying out: should reuse cached buffer to avoid extra allocs.
    out1 = your_dequantize_nf4(mlp.up_proj, use_custom_asm=True, use_cache_eviction=False)
    out2 = your_dequantize_nf4(mlp.up_proj, use_custom_asm=True, use_cache_eviction=False)
    assert out1.data_ptr() == out2.data_ptr()
    assert out1.shape == out2.shape
