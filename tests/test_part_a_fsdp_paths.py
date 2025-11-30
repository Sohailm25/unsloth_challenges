# ABOUTME: Tests helper paths for Part A NF4 kernel integration under FSDP2.
# ABOUTME: Verifies registry, gather fallback, and scatter-to-full behaviors.

import os
import sys
import types
import importlib.machinery
import torch

# Ensure project root on sys.path for direct module imports
ROOT_DIR = os.path.abspath(os.getcwd())
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Provide lightweight Triton stubs so we can import the module without GPU deps.
if "triton" not in sys.modules:
    triton_language = types.ModuleType("triton.language")
    triton_language.constexpr = object
    triton_language.float16 = "float16"
    triton_language.bfloat16 = "bfloat16"
    # Provide placeholder ops used only when kernel executes (not in these tests)
    def _noop(*args, **kwargs):
        return None
    triton_language.arange = triton_language.program_id = _noop
    triton_language.load = triton_language.store = triton_language.reshape = _noop
    triton_language.int32 = triton_language.uint32 = "int32"

    triton_mod = types.ModuleType("triton")
    triton_mod.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
    triton_mod.autotune = lambda *args, **kwargs: (lambda fn: fn)
    triton_mod.jit = lambda fn: fn
    triton_mod.Config = lambda *args, **kwargs: None
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.language = triton_language
    triton_language.__spec__ = importlib.machinery.ModuleSpec("triton.language", loader=None)

    triton_ops = types.ModuleType("triton.ops")
    triton_ops.__spec__ = importlib.machinery.ModuleSpec("triton.ops", loader=None)
    triton_matmul = types.ModuleType("triton.ops.matmul_perf_model")
    triton_matmul.__spec__ = importlib.machinery.ModuleSpec("triton.ops.matmul_perf_model", loader=None)
    triton_matmul.early_config_prune = lambda *args, **kwargs: None
    triton_matmul.estimate_matmul_time = lambda *args, **kwargs: 0
    triton_ops.matmul_perf_model = triton_matmul

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = triton_language
    sys.modules["triton.ops"] = triton_ops
    sys.modules["triton.ops.matmul_perf_model"] = triton_matmul

    # Stub bitsandbytes triton utils to report no Triton support so BnB skips kernels.
    bnb_triton_pkg = types.ModuleType("bitsandbytes.triton")
    bnb_triton_pkg.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton", loader=None)
    bnb_triton_pkg.__path__ = []  # mark as package
    bnb_triton_utils = types.ModuleType("bitsandbytes.triton.triton_utils")
    bnb_triton_utils.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.triton_utils", loader=None)
    bnb_triton_utils.is_triton_available = lambda: False
    bnb_deq_rowwise = types.ModuleType("bitsandbytes.triton.dequantize_rowwise")
    bnb_deq_rowwise.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.dequantize_rowwise", loader=None)
    bnb_deq_rowwise.dequantize_rowwise = lambda *args, **kwargs: None
    bnb_quant_rowwise = types.ModuleType("bitsandbytes.triton.quantize_rowwise")
    bnb_quant_rowwise.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.quantize_rowwise", loader=None)
    bnb_quant_rowwise.quantize_rowwise = lambda *args, **kwargs: (None, None, None)
    bnb_quant_col_T = types.ModuleType("bitsandbytes.triton.quantize_columnwise_and_transpose")
    bnb_quant_col_T.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.quantize_columnwise_and_transpose", loader=None)
    bnb_quant_col_T.quantize_columnwise_and_transpose = lambda *args, **kwargs: (None, None)
    bnb_quant_global = types.ModuleType("bitsandbytes.triton.quantize_global")
    bnb_quant_global.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.quantize_global", loader=None)
    bnb_quant_global.quantize_global = lambda *args, **kwargs: (None, None)
    bnb_quant_global.quantize_global_transpose = lambda *args, **kwargs: (None, None)
    bnb_int8 = types.ModuleType("bitsandbytes.triton.int8_matmul_rowwise_dequantize")
    bnb_int8.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.int8_matmul_rowwise_dequantize", loader=None)
    bnb_int8.int8_matmul_rowwise_dequantize = lambda *args, **kwargs: None
    bnb_int8_mixed = types.ModuleType("bitsandbytes.triton.int8_matmul_mixed_dequantize")
    bnb_int8_mixed.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes.triton.int8_matmul_mixed_dequantize", loader=None)
    bnb_int8_mixed.int8_matmul_mixed_dequantize = lambda *args, **kwargs: None
    sys.modules["bitsandbytes.triton"] = bnb_triton_pkg
    sys.modules["bitsandbytes.triton.triton_utils"] = bnb_triton_utils
    sys.modules["bitsandbytes.triton.dequantize_rowwise"] = bnb_deq_rowwise
    sys.modules["bitsandbytes.triton.quantize_rowwise"] = bnb_quant_rowwise
    sys.modules["bitsandbytes.triton.quantize_columnwise_and_transpose"] = bnb_quant_col_T
    sys.modules["bitsandbytes.triton.quantize_global"] = bnb_quant_global
    sys.modules["bitsandbytes.triton.int8_matmul_rowwise_dequantize"] = bnb_int8
    sys.modules["bitsandbytes.triton.int8_matmul_mixed_dequantize"] = bnb_int8_mixed

# Stub unsloth and peft integration modules to avoid import errors.
unsloth_utils = types.ModuleType("unsloth.kernels.utils")
unsloth_utils.fast_dequantize = lambda *args, **kwargs: None
unsloth_kernels = types.ModuleType("unsloth.kernels")
unsloth_kernels.utils = unsloth_utils
unsloth_mod = types.ModuleType("unsloth")
unsloth_mod.kernels = unsloth_kernels
sys.modules.setdefault("unsloth", unsloth_mod)
sys.modules.setdefault("unsloth.kernels", unsloth_kernels)
sys.modules.setdefault("unsloth.kernels.utils", unsloth_utils)

peft_integrations = types.ModuleType("peft.utils.integrations")
peft_integrations.dequantize_module_weight = lambda *args, **kwargs: None
peft_utils = types.ModuleType("peft.utils")
peft_utils.integrations = peft_integrations
peft_mod = types.ModuleType("peft")
peft_mod.utils = peft_utils
sys.modules.setdefault("peft", peft_mod)
sys.modules.setdefault("peft.utils", peft_utils)
sys.modules.setdefault("peft.utils.integrations", peft_integrations)

from challenge_b_train_with_part_a import (
    _QSTATE_OWNER,
    register_quant_modules,
    _reacquire_full_packed_from_module,
    _all_gather_packed,
    _scatter_local_to_full,
    _set_cached_full_packed,
    _get_cached_full_packed,
    _ENABLE_GATHER_CACHE,
    _infer_local_2d_shape,
    _decide_shard_axis,
)


class _DummyModule(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        # Simple quant_state marker object
        self.weight.quant_state = types.SimpleNamespace()


def test_register_and_reacquire_returns_weight():
    _QSTATE_OWNER.clear()
    weight = torch.tensor([1, 2, 3], dtype=torch.uint8)
    mod = _DummyModule(weight)
    register_quant_modules(mod)

    qs = mod.weight.quant_state
    result = _reacquire_full_packed_from_module(qs)
    assert torch.equal(result, weight)


def test_all_gather_returns_none_when_not_initialized():
    # torch.distributed.is_initialized() is False by default in tests
    weight = torch.tensor([0, 1], dtype=torch.uint8)
    gathered = _all_gather_packed(weight, full_packed_bytes=4)
    assert gathered is None


def test_cached_full_packed_roundtrip():
    qs = types.SimpleNamespace()
    tensor = torch.ones(4, dtype=torch.uint8)
    _set_cached_full_packed(qs, tensor)
    got = _get_cached_full_packed(qs, 4, tensor.device)
    assert got is tensor


def test_gather_cache_enabled_by_default():
    assert _ENABLE_GATHER_CACHE is False


def test_scatter_local_to_full_places_rows_correctly():
    # Shape (4, 2), world_size=2 -> local_rows=2
    qs = types.SimpleNamespace(shape=(4, 2))
    rank = 1
    world = 2
    # local packed bytes: local_rows * cols / 2 = 2
    A_local = torch.zeros(2, dtype=torch.uint8)
    local_out = torch.ones((2, 2), dtype=torch.float32)

    full = _scatter_local_to_full(local_out, qs, A_local, rank, world, (4, 2), False)
    assert full.shape == torch.Size([4, 2])
    assert torch.equal(full[:2], torch.zeros_like(full[:2]))
    assert torch.equal(full[2:], local_out)


def test_scatter_accepts_flat_local_out():
    qs = types.SimpleNamespace(shape=(4, 2))
    rank = 0
    world = 2
    A_local = torch.zeros(2, dtype=torch.uint8)
    flat_local = torch.ones(4, dtype=torch.float32)  # should reshape to (2, 2)

    full = _scatter_local_to_full(flat_local, qs, A_local, rank, world, (4, 2), False)
    assert full.shape == torch.Size([4, 2])
    assert torch.equal(full[:2], torch.ones_like(full[:2]))
    assert torch.equal(full[2:], torch.zeros_like(full[2:]))


def test_scatter_slices_when_kernel_returns_full():
    qs = types.SimpleNamespace(shape=(4, 2))
    rank = 1
    world = 2
    A_local = torch.zeros(2, dtype=torch.uint8)
    # Simulate kernel returning full shape even on shard
    full_local = torch.arange(8, dtype=torch.float32).view(4, 2)

    full = _scatter_local_to_full(full_local, qs, A_local, rank, world, (4, 2), False)
    assert full.shape == torch.Size([4, 2])
    # Rank1 should keep rows 2:4 from full_local
    assert torch.equal(full[:2], torch.zeros_like(full[:2]))
    assert torch.equal(full[2:], full_local[2:])


def test_infer_local_shape_handles_column_shard():
    # Full weight shape reports 4x8, but packed bytes correspond to 4x3 slice (column shard only)
    qs = types.SimpleNamespace(shape=(4, 8))
    # 4 rows * 3 cols = 12 weights => 6 bytes for NF4 packing
    packed = torch.zeros(6, dtype=torch.uint8)

    local_shape = _infer_local_2d_shape(packed, qs)

    assert local_shape == (4, 3)


def test_decide_shard_axis_prefers_input_mismatch_for_columns():
    weight_shape = (4, 3)
    input_shape = (2, 6)
    full_shape = (4, 6)
    local_shape = (4, 3)

    axis = _decide_shard_axis(weight_shape, input_shape, full_shape, local_shape)

    assert axis == "col"


def test_decide_shard_axis_row_when_rows_shrink():
    weight_shape = (2, 6)
    input_shape = (2, 6)
    full_shape = (4, 6)
    local_shape = (2, 6)

    axis = _decide_shard_axis(weight_shape, input_shape, full_shape, local_shape)

    assert axis == "row"


def test_matmul_pads_local_weight_columns(monkeypatch):
    import bitsandbytes.autograd._functions as bnb_autograd
    import bitsandbytes.functional as bnb_F
    import challenge_b_train_with_part_a as parta

    orig_forward = bnb_autograd.MatMul4Bit.forward
    orig_backward = bnb_autograd.MatMul4Bit.backward
    orig_dequant = bnb_F.dequantize_4bit

    stub_weight = torch.ones((2, 2), dtype=torch.float32)

    def fake_dequant(_B, _qs, *args, **kwargs):
        return stub_weight

    monkeypatch.setattr(parta, "patched_dequantize_4bit", fake_dequant)

    try:
        parta.enable_part_a_kernel()

        quant_state = types.SimpleNamespace(shape=(4, 4))
        packed = torch.empty(4, dtype=torch.uint8)  # 2 rows shard of 4x4 -> 4 bytes
        A = torch.randn(1, 4)
        ctx = types.SimpleNamespace(save_for_backward=lambda *args, **kwargs: None)

        out = bnb_autograd.MatMul4Bit.forward(ctx, A, packed, None, None, quant_state)

        expected = torch.nn.functional.linear(A, torch.nn.functional.pad(stub_weight, (0, 2)))
        assert out.shape == expected.shape
        assert torch.allclose(out, expected, atol=1e-6)
        assert ctx.local_cols == 4
    finally:
        bnb_autograd.MatMul4Bit.forward = orig_forward
        bnb_autograd.MatMul4Bit.backward = orig_backward
        bnb_F.dequantize_4bit = orig_dequant
        parta._USE_PART_A_KERNEL = False
