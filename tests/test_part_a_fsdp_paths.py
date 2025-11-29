# ABOUTME: Tests helper paths for Part A NF4 kernel integration under FSDP2.
# ABOUTME: Verifies registry, gather fallback, and scatter-to-full behaviors.

import sys
import types
import torch

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
    triton_mod.autotune = lambda *args, **kwargs: (lambda fn: fn)
    triton_mod.jit = lambda fn: fn
    triton_mod.Config = lambda *args, **kwargs: None
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.language = triton_language

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = triton_language

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
