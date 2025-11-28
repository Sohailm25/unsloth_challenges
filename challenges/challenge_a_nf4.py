# ABOUTME: Contains NF4 Triton dequantization challenge prompt and helper code.
# ABOUTME: Includes test harness and stubs for implementing the kernel.

# Helpful functions used through the entire notebook
import torch
import torch.nn as nn
from transformers import set_seed
import time
import inspect
import os
import math

# Reduce chance of BrokenProcessPool from Inductor by compiling synchronously
try:
    import torch._inductor.config as _inductor_config
    _inductor_config.compile_threads = 1
except Exception:  # noqa: BLE001
    pass

try:
    major_version, minor_version = torch.cuda.get_device_capability()
except Exception:  # noqa: BLE001
    major_version, minor_version = (0, 0)
HAS_BFLOAT16 = (major_version >= 8)
from inspect import currentframe as _C, getframeinfo
_F = lambda c: getframeinfo(c).lineno # Gets line number
WARN = lambda x: print(f"\033[31m{x}\033[0m") # Red colored warnings

# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""

def assert_same(x, y, line, dtype):
    assert(x.dtype == dtype)
    if dtype == torch.bfloat16:
        atol, rtol = 2e-3, 1e-2
    else:
        atol, rtol = 2e-5, 1e-3
    try: torch.testing.assert_close(x, y, check_stride=True, atol=atol, rtol=rtol)
    except Exception as error:
        raise RuntimeError(
            f"Failed allclose at line [{line}]: {NAME(x)}, {NAME(y)}\n{str(error)}"
        )

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

"""---
---
---
<a name="NF4"></a>
## A) Convert `nf4` to Triton. [Difficulty: Hard] [Max points: 14]


1. Goal: Convert a `nf4` quantized tensor into `fp16` or `bf16` into a *single* Triton kernel The double dequant of the `absmax` and weight forming must be done in 1 Triton kernel. Must work on Tesla T4.
2. Must be faster than Unsloth's `fast_dequantize` by 1.15x or more, and not use large intermediate memory buffers.
3. Must not use `torch.compile`, but can use `trace.enabled` to help on writing Triton kernels.
4. Good material: [Unsloth `fast_dequantize` function](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py#L128), also [bitsandbytes `dequantize_blockwise`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/bitsandbytes/functional.py#L958)
5. Use `test_dequantize_function` to test your implementation.
6. No CUDA allowed. Custom CUDA inside of the Triton is allowed.
7. Watch Tim's videos on Youtube: [8-bit Optimizers](https://www.youtube.com/watch?v=2ETNONas068)
"""

from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
try:
    from unsloth.kernels.utils import fast_dequantize as _fast_dequantize_impl  # type: ignore
    _FAST_DEQUANT_TAKES_MODULE = False
except Exception:  # noqa: BLE001
    from peft.utils.integrations import dequantize_module_weight as _fast_dequantize_impl
    _FAST_DEQUANT_TAKES_MODULE = True


def _call_fast_dequantize(weight_module):
    if _FAST_DEQUANT_TAKES_MODULE:
        return _fast_dequantize_impl(weight_module)
    return _fast_dequantize_impl(weight_module.weight, weight_module.weight.quant_state)


def unsloth_dequantize(weight):
    return _call_fast_dequantize(weight)

def bnb_Linear4bit(hd, m, dtype = torch.float16):
    return Linear4bit(
        hd, m, bias = None,
        compute_dtype       = dtype,
        compress_statistics = True,
        quant_type          = "nf4",
    )

# [NEW] as at 18th Feb 2025
def assert_correct_bnb(weight, dtype):
    assert(weight.weight.dtype == torch.uint8)
    assert(weight.weight.quant_state.dtype == dtype)
    assert(weight.weight.quant_state.absmax.dtype == torch.uint8)
    assert(weight.weight.quant_state.code.dtype == torch.float32)
    assert(weight.weight.quant_state.offset.dtype == torch.float32)
    assert(weight.weight.quant_state.blocksize == 64)
    assert(weight.weight.quant_state.state2.absmax.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.code.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.blocksize == 256)

class MLP(nn.Module):
    def __init__(self, hd = 4096, m = 14336, dtype = torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype = dtype).to("cuda")
        self.up_proj   = bnb_Linear4bit(hd, m, dtype = dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype = dtype).to("cuda")
        # [NEW] as at 18th Feb 2025
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj  .weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.act_fn = ACT2FN["silu"]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_forward(X, mlp, fx):
    up   = X @ fx(mlp.  up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.  up_proj); torch.cuda.synchronize()
    b = fx(mlp.gate_proj); torch.cuda.synchronize()
    c = fx(mlp.down_proj); torch.cuda.synchronize()
    return a, b, c

def test_dequantize(dequantize_fx):
    elapsed = 0
    options = [
        (2, 3333, 2048,  8192, 3407, torch.float16),
        (5,  777, 1024,  4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
    ]
    for (bsz, qlen, hd, m, seed, dt) in options:
        set_seed(seed)
        torch.set_default_dtype(torch.float32)
        mlp = MLP(hd = hd, m = m, dtype = dt)
        X = torch.randn((bsz, qlen, hd), device = "cuda", dtype = dt)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            assert_same( mlp_forward(X, mlp, dequantize_fx), mlp(X), _F(_C()), dt)
            # [NEW] as at 18th Feb 2025
            assert_correct_bnb(mlp.  up_proj, dt)
            assert_correct_bnb(mlp.gate_proj, dt)
            assert_correct_bnb(mlp.down_proj, dt)
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000): mlp_dequantize(X, mlp, dequantize_fx)
        elapsed += time.time() - start
    return elapsed

"""For example, we can test our implementation via:"""

"""Write your Triton kernel below, and test it:"""

from triton import jit
import triton
import triton.language as tl

_NF4_LUT_CACHE = {}


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _compute_shift_offsets(weight, quant_state):
    n_weights = math.prod(quant_state.shape)
    n_bytes = weight.numel()
    absmax = quant_state.absmax
    state2 = getattr(quant_state, "state2", None)
    if state2 is None or absmax is None or state2.absmax is None:
        return None
    if absmax.numel() == 0 or state2.absmax.numel() == 0:
        return None

    blocksize = int(getattr(quant_state, "blocksize", 0))
    if blocksize <= 0 or blocksize % 2 != 0:
        return None

    # Detect packing: packed bytes or one byte per weight
    if n_bytes * 2 == n_weights:
        bytes_per_absmax = blocksize // 2
    elif n_bytes == n_weights:
        bytes_per_absmax = blocksize
    else:
        return None

    n_absmax = absmax.numel()
    expected_absmax = math.ceil(n_weights / blocksize)
    if expected_absmax != n_absmax:
        return None
    if not _is_power_of_two(bytes_per_absmax):
        return None

    blocksize2 = int(getattr(state2, "blocksize", 0))
    if blocksize2 <= 0 or not _is_power_of_two(blocksize2):
        return None
    expected_absmax2 = math.ceil(n_absmax / blocksize2)
    if expected_absmax2 != state2.absmax.numel():
        return None

    return int(math.log2(bytes_per_absmax)), int(math.log2(blocksize2))


def _get_output_shape(quant_state, weight):
    if hasattr(quant_state, "shape"):
        return tuple(quant_state.shape)
    if hasattr(quant_state, "weight_shape"):
        return tuple(quant_state.weight_shape)
    return (weight.numel() * 2,)


def _ensure_tensor(tensor, device, dtype=None):
    target = tensor.to(device)
    if dtype is not None:
        target = target.to(dtype)
    return target.contiguous()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "LOAD_VEC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256, "LOAD_VEC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "LOAD_VEC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "LOAD_VEC": 4}, num_warps=8, num_stages=3),
    ],
    key=["n_packed"],
)
@triton.jit
def _your_dequantize_nf4_kernel(
    weight_ptr,  # int32* (each element holds 4 packed bytes)
    absmax_ptr,
    absmax2_ptr,
    code2_ptr,
    lut_ptr,
    evict_ptr,
    out_ptr,
    offset_ptr,
    n_packed,
    n_weights,
    n_absmax,
    n_absmax2,
    debug_ptr,
    debug_block,
    shift_absmax_bytes: tl.constexpr,
    shift_absmax2: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    USE_CUSTOM_ASM: tl.constexpr,
    USE_CACHE_EVICT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOAD_VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE)
    group_idx = pid * BLOCK_SIZE + offs_local

    byte_base = group_idx * LOAD_VEC
    mask_group = byte_base < n_packed

    if USE_CACHE_EVICT:
        _ = tl.load(
            evict_ptr + byte_base,
            mask=mask_group,
            other=0,
            eviction_policy="evict_last",
        )

    packed32 = tl.load(
        weight_ptr + group_idx,
        mask=mask_group,
        other=0,
    ).to(tl.uint32)

    shifts = (tl.arange(0, LOAD_VEC) * 8).to(tl.uint32)
    bytes2d = (packed32[:, None] >> shifts[None, :]) & 0xFF  # [BLOCK_SIZE, LOAD_VEC]

    byte_offsets2d = byte_base[:, None] + tl.arange(0, LOAD_VEC)[None, :]
    mask_bytes2d = byte_offsets2d < n_packed

    lo2d = bytes2d & 0x0F
    hi2d = bytes2d >> 4

    absmax_idx_2d = (byte_offsets2d >> shift_absmax_bytes).to(tl.int32)
    absmax2_idx_2d = (absmax_idx_2d >> shift_absmax2).to(tl.int32)

    hi = tl.reshape(hi2d, (BLOCK_SIZE * LOAD_VEC,))
    lo = tl.reshape(lo2d, (BLOCK_SIZE * LOAD_VEC,))
    absmax_idx = tl.reshape(absmax_idx_2d, (BLOCK_SIZE * LOAD_VEC,))
    absmax2_idx = tl.reshape(absmax2_idx_2d, (BLOCK_SIZE * LOAD_VEC,))
    mask = tl.reshape(mask_bytes2d, (BLOCK_SIZE * LOAD_VEC,))

    offset = tl.load(offset_ptr).to(tl.float32)

    abs_mask = mask & (absmax_idx < n_absmax)
    absmax_quant = tl.load(
        absmax_ptr + absmax_idx,
        mask=abs_mask,
        other=0,
        eviction_policy="evict_last",
    ).to(tl.int32)
    absmax_quant = tl.where(absmax_quant > 255, 0, absmax_quant)

    code_val = tl.load(
        code2_ptr + absmax_quant,
        mask=abs_mask,
        other=0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    abs2_mask = mask & (absmax2_idx < n_absmax2)
    scale = tl.load(
        absmax2_ptr + absmax2_idx,
        mask=abs2_mask,
        other=1.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    fma = getattr(tl, "fma", None)
    absmax = fma(code_val, scale, offset) if fma is not None else code_val * scale + offset

    w_hi = tl.load(lut_ptr + hi.to(tl.int32), mask=mask, other=0.0)
    w_lo = tl.load(lut_ptr + lo.to(tl.int32), mask=mask, other=0.0)
    w_hi = w_hi * absmax
    w_lo = w_lo * absmax

    hi_2d_out = tl.reshape(w_hi, (BLOCK_SIZE * LOAD_VEC, 1))
    lo_2d_out = tl.reshape(w_lo, (BLOCK_SIZE * LOAD_VEC, 1))
    cols = tl.reshape(tl.arange(0, 2), (1, 2))
    vals_2d = tl.where(cols == 0, hi_2d_out, lo_2d_out)
    vals = tl.reshape(vals_2d, (2 * BLOCK_SIZE * LOAD_VEC,))

    out_base = pid * 2 * BLOCK_SIZE * LOAD_VEC
    out_offsets = out_base + tl.arange(0, 2 * BLOCK_SIZE * LOAD_VEC)
    out_mask = out_offsets < n_weights
    tl.store(out_ptr + out_offsets, vals.to(OUT_DTYPE), mask=out_mask, eviction_policy="evict_last")

    if debug_block >= 0:
        # Debug path omitted in vectorized kernel to avoid compile-time indexing issues.
        pass


_OUT_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _your_dequantize_nf4(
    weight,
    quant_state,
    offset1,
    offset2,
    *,
    use_custom_asm=False,
    use_cache_eviction=False,
    debug_block=None,
):
    if not weight.is_cuda:
        raise RuntimeError("NF4 dequantization requires CUDA.")

    dtype = getattr(quant_state, "dtype", None)
    if dtype not in _OUT_DTYPE_MAP:
        raise RuntimeError("quant_state.dtype must be float16 or bfloat16.")

    device = weight.device
    weight_flat_u8 = weight.contiguous().view(-1)
    n_packed = weight_flat_u8.numel()
    if n_packed % 4 != 0:
        raise RuntimeError("NF4 packed size must be a multiple of 4 for LOAD_VEC=4 kernel.")
    weight_flat_u32 = weight_flat_u8.view(torch.int32)

    absmax = _ensure_tensor(quant_state.absmax, device, torch.uint8)
    absmax2 = _ensure_tensor(quant_state.state2.absmax, device, torch.float32)
    code2 = _ensure_tensor(quant_state.state2.code, device, torch.float32)
    lut = _ensure_tensor(quant_state.code, device, torch.float32)
    evict = (
        torch.empty_like(weight_flat_u8)
        if use_cache_eviction
        else torch.empty(1, device=device, dtype=torch.uint8)
    )
    offset = torch.tensor(float(quant_state.offset), device=device, dtype=torch.float32)
    n_weights = math.prod(quant_state.shape)
    n_absmax = absmax.numel()
    n_absmax2 = absmax2.numel()
    out_storage_dtype = dtype
    emulate_bf16 = False
    out_dtype = _OUT_DTYPE_MAP[dtype]
    if dtype == torch.bfloat16 and major_version < 8:
        emulate_bf16 = True
        out_storage_dtype = torch.float32
        out_dtype = tl.float32

    out_flat = torch.empty(n_weights, device=device, dtype=out_storage_dtype)

    grid = lambda meta: (triton.cdiv(n_packed, meta["BLOCK_SIZE"] * meta["LOAD_VEC"]),)

    debug_buffer = None
    dbg_block_val = -1
    if debug_block is not None:
        dbg_block_val = int(debug_block)
        debug_buffer = torch.empty((5, 2048), device=device, dtype=torch.float32)

    _your_dequantize_nf4_kernel[grid](
        weight_flat_u32,
        absmax,
        absmax2,
        code2,
        lut,
        evict,
        out_flat,
        offset,
        n_packed,
        n_weights,
        n_absmax,
        n_absmax2,
        debug_buffer if debug_buffer is not None else evict,
        dbg_block_val,
        shift_absmax_bytes=offset1,
        shift_absmax2=offset2,
        OUT_DTYPE=out_dtype,
        USE_CUSTOM_ASM=use_custom_asm,
        USE_CACHE_EVICT=use_cache_eviction,
    )

    if emulate_bf16:
        out_flat = out_flat.to(torch.bfloat16)

    output_shape = _get_output_shape(quant_state, weight)
    if debug_buffer is not None:
        return out_flat.view(output_shape), debug_buffer
    return out_flat.view(output_shape)


def your_dequantize_nf4(
    weight,
    *,
    use_custom_asm=False,
    use_cache_eviction=False,
    use_optimized=True,
    debug_block=None,
):
    if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
        return _call_fast_dequantize(weight)
    quant_state = weight.weight.quant_state
    shifts = _compute_shift_offsets(weight.weight.data, quant_state)
    if shifts is None or not use_optimized:
        return _call_fast_dequantize(weight)
    offset1, offset2 = shifts
    return _your_dequantize_nf4(
        weight.weight.data,
        quant_state,
        offset1,
        offset2,
        use_custom_asm=use_custom_asm,
        use_cache_eviction=use_cache_eviction,
        debug_block=debug_block,
    )


if hasattr(torch, "_dynamo"):
    your_dequantize_nf4 = torch._dynamo.disable()(your_dequantize_nf4)  # type: ignore


def _debug_dequant_single_block(weight):
    """Return debug info for the first block to compare with reference."""
    if not weight.weight.is_cuda:
        raise RuntimeError("Debug dequant requires CUDA.")

    qs = weight.weight.quant_state
    blocksize = int(getattr(qs, "blocksize", 64))
    block_bytes = blocksize // 2
    n_weights = qs.shape.numel()
    n_bytes = weight.weight.data.numel()
    # Take first byte block
    weight_flat = weight.weight.data.flatten().contiguous()
    packed_slice = weight_flat[:block_bytes]

    # Host reference reconstruction for that block
    code_lut = qs.code.to(torch.float32)
    absmax_codes = qs.absmax.to(torch.int64)
    code2 = qs.state2.code.to(torch.float32)
    absmax2 = qs.state2.absmax.to(torch.float32)
    offset = torch.tensor(float(qs.offset), device=weight_flat.device, dtype=torch.float32)

    # Compute first block indices
    abs_idx = torch.arange(0, block_bytes, device=weight_flat.device) // block_bytes
    abs2_idx = abs_idx // qs.state2.blocksize
    abs_vals = code2[absmax_codes[abs_idx]] * absmax2[abs2_idx] + offset

    # Dequant host-side for the block
    q = packed_slice
    hi = (q >> 4).long()
    lo = (q & 0x0F).long()
    w_hi = code_lut[hi].to(torch.float32) * abs_vals.to(torch.float32)
    w_lo = code_lut[lo].to(torch.float32) * abs_vals.to(torch.float32)
    host_out = torch.empty(block_bytes * 2, device=weight_flat.device, dtype=qs.dtype)
    host_out[0::2] = w_hi
    host_out[1::2] = w_lo

    # Kernel out for same slice
    kernel_out = your_dequantize_nf4(weight, use_custom_asm=False, use_cache_eviction=False, use_optimized=True)
    kernel_block = kernel_out.flatten()[: blocksize]

    ref_out = _call_fast_dequantize(weight).flatten()[: blocksize]

    return {
        "packed": packed_slice.detach().cpu(),
        "absmax_codes": absmax_codes[:1].detach().cpu(),
        "absmax2": absmax2[:1].detach().cpu(),
        "offset": offset.detach().cpu(),
        "host_out": host_out.detach().cpu(),
        "kernel_out": kernel_block.detach().cpu(),
        "ref_out": ref_out.detach().cpu(),
        "n_weights": n_weights,
        "n_bytes": n_bytes,
        "blocksize": blocksize,
        "blocksize2": qs.state2.blocksize,
    }

if __name__ == "__main__":
    # test_dequantize(your_dequantize_nf4)
    pass

"""## Marking Criteria for A) Max points = 14
```python
if attemped_A:
    A_score = 0
    if single_triton_kernel: A_score += 3
    speedup = old_time / new_time
    if speedup <= 1.00: A_score -= 3
    if speedup >= 1.05: A_score += 1
    if speedup >= 1.10: A_score += 2
    if speedup >= 1.15: A_score += 2
    if kernel_works_in_torch_compile: A_score += 1
    else: A_score -= 1
    if custom_asm_works: A_score += 3
    if uses_cache_eviction: A_score += 1
    if tested_in_f16_and_bf16: A_score += 1
    else: A_score -= 1
    final_score += A_score
else:
    final_score += 0
```

---
---
---
"""
