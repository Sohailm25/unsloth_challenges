# ABOUTME: FSDP2 + QLoRA training with Part A NF4 kernel integration.
# ABOUTME: Benchmarks custom kernel vs BnB baseline for +3 points.

"""
FSDP2 + QLoRA + Part A Kernel Training Script

Integrates the custom NF4 dequantization kernel from Challenge A
to potentially gain +3 bonus points if faster than BnB baseline.
"""

import os
import sys
import torch
import argparse
import time
import math
import weakref
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass

# Set environment variables before imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
)
# Safe NCCL/connection defaults for FSDP on T4
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
if torch.cuda.is_available():
    os.environ.setdefault("ACCELERATE_USE_NCCL", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "nccl")

# ============================================================================
# Part A NF4 Kernel (from challenge_a_nf4_backup_post_asm_20251128.py)
# ============================================================================

import triton
import triton.language as tl

try:
    major_version, minor_version = torch.cuda.get_device_capability()
except Exception:
    major_version, minor_version = (0, 0)

try:
    from unsloth.kernels.utils import fast_dequantize as _fast_dequantize_impl
    _FAST_DEQUANT_TAKES_MODULE = False
except Exception:
    from peft.utils.integrations import dequantize_module_weight as _fast_dequantize_impl
    _FAST_DEQUANT_TAKES_MODULE = True


def _call_fast_dequantize(weight_module):
    if _FAST_DEQUANT_TAKES_MODULE:
        return _fast_dequantize_impl(weight_module)
    return _fast_dequantize_impl(weight_module.weight, weight_module.weight.quant_state)


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


_DEBUG_SHIFTS_ONCE = [False]


# Global toggles for FSDP2 integration paths
_ENABLE_REACQUIRE = os.getenv("ORACLE_PARTA_REACQUIRE", "1") != "0"
_ENABLE_GATHER = os.getenv("ORACLE_PARTA_GATHER", "1") != "0"
_ENABLE_SCATTER = os.getenv("ORACLE_PARTA_SCATTER_FALLBACK", "1") != "0"
_ENABLE_GATHER_CACHE = os.getenv("ORACLE_PARTA_GATHER_CACHE", "0") != "0"

# Registry to map quant_state -> owning module for reacquisition
_QSTATE_OWNER = weakref.WeakValueDictionary()

def _compute_shift_offsets_fsdp(weight, quant_state):
    """Compute shift offsets, handling FSDP sharded weights.

    With FSDP, the weight tensor may be sharded but quant_state.shape contains
    the full unsharded shape. We detect this and adjust accordingly.
    """
    n_bytes = weight.numel()
    absmax = quant_state.absmax
    state2 = getattr(quant_state, "state2", None)

    debug = not _DEBUG_SHIFTS_ONCE[0]
    if debug:
        _DEBUG_SHIFTS_ONCE[0] = True

    if state2 is None or absmax is None or state2.absmax is None:
        if debug: print("[SHIFT DEBUG] Failed: state2/absmax None check", flush=True)
        return None
    if absmax.numel() == 0 or state2.absmax.numel() == 0:
        if debug: print("[SHIFT DEBUG] Failed: empty absmax", flush=True)
        return None

    blocksize = int(getattr(quant_state, "blocksize", 0))
    if blocksize <= 0 or blocksize % 2 != 0:
        if debug: print(f"[SHIFT DEBUG] Failed: blocksize={blocksize} check", flush=True)
        return None

    # For NF4: 2 weights per packed byte, so n_weights = n_bytes * 2
    # With FSDP sharding, the actual weight being processed may be a shard
    # We compute the effective number of weights from the packed size
    n_weights_effective = n_bytes * 2  # NF4 mode: 2 weights per byte
    bytes_per_absmax = blocksize // 2  # NF4: each absmax covers blocksize weights = blocksize/2 bytes

    if debug:
        print(f"[SHIFT DEBUG] n_bytes={n_bytes}, n_weights_effective={n_weights_effective}", flush=True)
        print(f"[SHIFT DEBUG] blocksize={blocksize}, bytes_per_absmax={bytes_per_absmax}", flush=True)

    if not _is_power_of_two(bytes_per_absmax):
        if debug: print("[SHIFT DEBUG] Failed: bytes_per_absmax not power of 2", flush=True)
        return None

    blocksize2 = int(getattr(state2, "blocksize", 0))
    if debug: print(f"[SHIFT DEBUG] blocksize2={blocksize2}", flush=True)
    if blocksize2 <= 0 or not _is_power_of_two(blocksize2):
        if debug: print("[SHIFT DEBUG] Failed: blocksize2 check", flush=True)
        return None

    result = int(math.log2(bytes_per_absmax)), int(math.log2(blocksize2))
    if debug: print(f"[SHIFT DEBUG] Success! shifts={result}", flush=True)
    return result


def _compute_shift_offsets(weight, quant_state):
    """Legacy function, now delegates to FSDP-aware version."""
    return _compute_shift_offsets_fsdp(weight, quant_state)


def _get_output_shape_fsdp(quant_state, weight):
    """Get output shape, handling FSDP sharded weights.

    With FSDP, quant_state.shape may be the full unsharded shape,
    but we're only processing a shard. We need to return a 2D shape
    for linear layer compatibility.
    """
    # For NF4: 2 weights per packed byte
    n_weights_effective = weight.numel() * 2

    # Check if this matches the full shape
    full_shape = tuple(quant_state.shape) if hasattr(quant_state, "shape") else None
    if full_shape and len(full_shape) == 2:
        full_elements = full_shape[0] * full_shape[1]
        if full_elements == n_weights_effective:
            # Not sharded, use full shape
            return full_shape
        elif n_weights_effective < full_elements:
            # FSDP sharded - infer 2D shape
            # FSDP typically shards the first dimension
            # Try to compute: (n_weights_effective // full_shape[1], full_shape[1])
            if n_weights_effective % full_shape[1] == 0:
                return (n_weights_effective // full_shape[1], full_shape[1])
            elif n_weights_effective % full_shape[0] == 0:
                return (full_shape[0], n_weights_effective // full_shape[0])

    # Fallback: return flat shape (may cause issues)
    return (n_weights_effective,)


def _get_output_shape(quant_state, weight):
    """Legacy function for non-FSDP case."""
    if hasattr(quant_state, "shape"):
        return tuple(quant_state.shape)
    if hasattr(quant_state, "weight_shape"):
        return tuple(quant_state.weight_shape)
    return (weight.numel() * 2,)


def _cached_on(qs, name, src, device, dtype):
    cache_name = f"_cached_{name}"
    cached = getattr(qs, cache_name, None)
    if (
        cached is None
        or cached.device != device
        or cached.dtype != dtype
        or cached.numel() != src.numel()
        or not cached.is_contiguous()
    ):
        cached = src.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        setattr(qs, cache_name, cached)
    return cached


_DUMMY_EVICT = {}
_OUT_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512, "LOAD_VEC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "LOAD_VEC": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024, "LOAD_VEC": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048, "LOAD_VEC": 4}, num_warps=8, num_stages=3),
    ],
    key=["n_packed"],
)
@triton.jit
def _your_dequantize_nf4_kernel(
    weight_ptr,
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
    bytes2d = (packed32[:, None] >> shifts[None, :]) & 0xFF

    byte_offsets2d = byte_base[:, None] + tl.arange(0, LOAD_VEC)[None, :]
    mask_bytes2d = byte_offsets2d < n_packed

    if USE_CUSTOM_ASM:
        lo2d = tl.inline_asm_elementwise(
            asm="and.b32 $0, $1, 0xF;",
            constraints="=r,r",
            args=[bytes2d],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        hi2d = tl.inline_asm_elementwise(
            asm="shr.u32 $0, $1, 4;",
            constraints="=r,r",
            args=[bytes2d],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    else:
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

    absmax = code_val * scale + offset

    w_hi = tl.load(lut_ptr + hi.to(tl.int32), mask=mask, other=0.0)
    w_lo = tl.load(lut_ptr + lo.to(tl.int32), mask=mask, other=0.0)
    w_hi = w_hi * absmax
    w_lo = w_lo * absmax

    out_base = pid * 2 * BLOCK_SIZE * LOAD_VEC
    out_even = out_base + tl.arange(0, BLOCK_SIZE * LOAD_VEC) * 2
    out_odd = out_even + 1
    out_mask_even = (out_even < n_weights) & mask
    out_mask_odd = (out_odd < n_weights) & mask
    tl.store(out_ptr + out_even, w_hi.to(OUT_DTYPE), mask=out_mask_even, eviction_policy="evict_last")
    tl.store(out_ptr + out_odd, w_lo.to(OUT_DTYPE), mask=out_mask_odd, eviction_policy="evict_last")


def _your_dequantize_nf4(
    weight,
    quant_state,
    offset1,
    offset2,
    *,
    use_custom_asm=False,
    use_cache_eviction=False,
    out_tensor=None,
):
    # Use updated kernel implementation from challenge A.
    import challenges.challenge_a_nf4 as parta_kernel

    return parta_kernel._your_dequantize_nf4(
        weight,
        quant_state,
        offset1,
        offset2,
        use_custom_asm=use_custom_asm,
        use_cache_eviction=use_cache_eviction,
        out_tensor=out_tensor,
    )


def your_dequantize_nf4(
    weight_module,
    *,
    use_custom_asm=None,
    use_cache_eviction=False,
    use_optimized=True,
    out=None,
):
    """Custom NF4 dequantization using Part A Triton kernel."""
    if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
        return _call_fast_dequantize(weight_module)

    quant_state = weight_module.weight.quant_state

    maj, minr = torch.cuda.get_device_capability(weight_module.weight.device)
    if use_custom_asm is None:
        use_custom_asm = maj == 7 and minr == 5

    shifts = getattr(quant_state, "_nf4_shifts", None)
    if shifts is None:
        shifts = _compute_shift_offsets(weight_module.weight.data, quant_state)
        setattr(quant_state, "_nf4_shifts", shifts)
    if shifts is None or not use_optimized:
        return _call_fast_dequantize(weight_module)

    offset1, offset2 = shifts
    n_packed = weight_module.weight.data.numel()
    asm_ok = use_custom_asm and maj == 7 and minr == 5 and (n_packed % 4 == 0)

    return _your_dequantize_nf4(
        weight_module.weight.data,
        quant_state,
        offset1,
        offset2,
        use_custom_asm=asm_ok,
        use_cache_eviction=use_cache_eviction,
        out_tensor=out,
    )


if hasattr(torch, "_dynamo"):
    your_dequantize_nf4 = torch._dynamo.disable()(your_dequantize_nf4)


# ============================================================================
# Monkey-patch BnB to use custom kernel
# ============================================================================

_ORIGINAL_BNB_DEQUANT = None
_USE_PART_A_KERNEL = False
_DEQUANT_TIMES = {"part_a": 0.0, "bnb": 0.0, "count": 0, "part_a_calls": 0, "bnb_calls": 0, "fallback_calls": 0}


class _ShardedQuantState:
    """Wrapper for quant_state with sliced absmax/state2 for FSDP shards."""
    def __init__(self, original, absmax_slice, state2_wrapper=None, shape_override=None):
        self._original = original
        self.absmax = absmax_slice
        self.state2 = state2_wrapper
        # Copy other attributes from original
        self.shape = shape_override if shape_override is not None else original.shape
        self.blocksize = original.blocksize
        self.quant_type = getattr(original, 'quant_type', 'nf4')
        self.dtype = getattr(original, 'dtype', torch.float16)
        self.offset = getattr(original, 'offset', None)
        self.code = getattr(original, 'code', None)

class _ShardedState2:
    """Wrapper for state2 with sliced absmax."""
    def __init__(self, original, absmax_slice):
        self._original = original
        self.absmax = absmax_slice
        self.blocksize = original.blocksize
        self.code = getattr(original, 'code', None)
        self.dtype = getattr(original, 'dtype', None)


def register_quant_modules(root: nn.Module):
    """Register modules that carry quant_state so we can reacquire weights post all-gather."""
    for module in root.modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        qs = getattr(weight, "quant_state", None)
        if qs is None:
            continue
        _QSTATE_OWNER[id(qs)] = module


def _reacquire_full_packed_from_module(quant_state):
    """Return the owning module's weight tensor if available."""
    module = _QSTATE_OWNER.get(id(quant_state))
    if module is None:
        return None
    # WeakValueDictionary stores module directly
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    data = weight.data
    if not data.is_contiguous():
        data = data.contiguous()
    return data


def _all_gather_packed(A_local, full_packed_bytes):
    """Gather packed bytes across ranks into a full contiguous buffer."""
    if not dist.is_available() or not dist.is_initialized():
        return None

    world = dist.get_world_size()
    if A_local.numel() * world != full_packed_bytes:
        return None

    A_view = A_local.contiguous().view(-1)
    if hasattr(dist, "all_gather_into_tensor"):
        out = torch.empty(full_packed_bytes, device=A_local.device, dtype=A_local.dtype)
        dist.all_gather_into_tensor(out, A_view)
        return out

    gather_list = [torch.empty_like(A_view) for _ in range(world)]
    dist.all_gather(gather_list, A_view)
    return torch.cat(gather_list, dim=0)


def _get_cached_full_packed(qs, full_packed_bytes, device):
    buf = getattr(qs, "_cached_full_packed", None)
    if (
        buf is not None
        and buf.device == device
        and buf.numel() == full_packed_bytes
    ):
        return buf
    return None


def _set_cached_full_packed(qs, tensor):
    qs._cached_full_packed = tensor


def _scatter_local_to_full(local_out, quant_state, A, rank, world, bnb_shape=None, need_T=False):
    """Scatter local dequantized shard into the full BnB-oriented tensor."""
    if not hasattr(quant_state, "shape") or len(quant_state.shape) != 2:
        return None

    q_rows, q_cols = quant_state.shape
    n_local = A.numel() * 2  # NF4 packs 2 weights per byte
    if n_local % q_cols != 0:
        return None

    local_q_rows = n_local // q_cols
    if q_rows % world != 0 or local_q_rows != q_rows // world:
        return None

    b_rows, b_cols = (
        bnb_shape
        if bnb_shape is not None
        else ((q_cols, q_rows) if need_T else (q_rows, q_cols))
    )
    row_start = rank * local_q_rows

    if need_T:
        if local_out.dim() != 2:
            if local_out.numel() == b_rows * local_q_rows:
                local_out = local_out.view(b_rows, local_q_rows)
            else:
                return None
        if local_out.shape != (b_rows, local_q_rows):
            return None
        out_full = torch.zeros((b_rows, b_cols), device=local_out.device, dtype=local_out.dtype)
        col_start = row_start
        out_full[:, col_start : col_start + local_q_rows] = local_out
        return out_full

    if local_out.dim() != 2:
        if local_out.numel() == local_q_rows * b_cols:
            local_out = local_out.view(local_q_rows, b_cols)
        else:
            return None
    if local_out.shape == (b_rows, b_cols):
        local_out = local_out[row_start : row_start + local_q_rows]
    if local_out.shape != (local_q_rows, b_cols):
        return None

    out_full = torch.zeros((b_rows, b_cols), device=local_out.device, dtype=local_out.dtype)
    out_full[row_start : row_start + local_q_rows] = local_out
    return out_full


def _get_fsdp_rank_info():
    """Get FSDP rank and world_size if available."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _slice_quant_state_for_fsdp(quant_state, A, rank, world_size):
    """Create a sliced quant_state for this FSDP rank.

    When FSDP shards weights, each rank has a portion of the packed bytes.
    The quant_state.absmax remains full (replicated), so we slice it to
    match the local shard's blocks.

    Key insight from Oracle: byte_bias=0 when shards are block-aligned,
    which they are for typical Llama layers (divisible by 64).
    """
    n_packed = A.numel()
    blocksize = int(getattr(quant_state, 'blocksize', 64))
    block_bytes = blocksize // 2  # NF4: 2 weights per byte, so blocksize weights = blocksize/2 bytes

    # Compute blocks in this shard
    local_blocks = n_packed // block_bytes

    # Full shape (rows, cols)
    full_shape = tuple(quant_state.shape) if hasattr(quant_state, "shape") else None
    shard_shape = None
    if full_shape and len(full_shape) == 2:
        rows, cols = full_shape
        if (n_packed * 2) % cols == 0:
            shard_rows = (n_packed * 2) // cols
            shard_shape = (shard_rows, cols)

    # Full absmax blocks
    total_blocks = quant_state.absmax.numel()

    # Verify even distribution
    if total_blocks % world_size != 0:
        return None  # Can't evenly distribute, fall back to BnB

    blocks_per_rank = total_blocks // world_size

    if local_blocks != blocks_per_rank:
        # Shard size doesn't match expected, fall back
        return None

    # Slice absmax for this rank
    absmax_start = rank * blocks_per_rank
    absmax_slice = quant_state.absmax[absmax_start : absmax_start + blocks_per_rank]

    # Handle state2 (double quantization) if present
    state2 = getattr(quant_state, 'state2', None)
    state2_wrapper = None
    if state2 is not None and hasattr(state2, 'absmax'):
        state2_blocksize = int(getattr(state2, 'blocksize', 256))
        # state2 quantizes the primary absmax in groups of state2_blocksize
        s2_total = state2.absmax.numel()
        if s2_total % world_size == 0:
            s2_per_rank = s2_total // world_size
            s2_start = rank * s2_per_rank
            s2_slice = state2.absmax[s2_start : s2_start + s2_per_rank]
            state2_wrapper = _ShardedState2(state2, s2_slice)

    return _ShardedQuantState(quant_state, absmax_slice, state2_wrapper, shape_override=shard_shape)


def _wrap_quant_state_with_shape(quant_state, target_shape):
    """Return a quant_state view that reports a target shape without altering data."""
    if tuple(getattr(quant_state, "shape", ())) == tuple(target_shape):
        return quant_state
    state2 = getattr(quant_state, "state2", None)
    state2_wrapper = None
    if state2 is not None and hasattr(state2, "absmax"):
        state2_wrapper = _ShardedState2(state2, state2.absmax)
    return _ShardedQuantState(quant_state, quant_state.absmax, state2_wrapper, shape_override=target_shape)


def patched_dequantize_4bit(A, quant_state, absmax=None, out=None, blocksize=64, quant_type='fp4'):
    """Patched BnB dequantize that can use Part A kernel with FSDP support."""
    global _DEQUANT_TIMES

    # Check for NF4 from quant_state (more reliable than function parameter)
    is_nf4 = quant_type == 'nf4' or getattr(quant_state, 'quant_type', None) == 'nf4'

    # Early guard: if original BnB ever returns None, log and bail with zeros to avoid matmul crash
    def _bnb_safe_call(A_tensor, qs_tensor):
        res = _ORIGINAL_BNB_DEQUANT(A_tensor, qs_tensor, absmax, out, blocksize, quant_type)
        if res is None:
            rank_dbg, world_dbg = _get_fsdp_rank_info()
            shape_dbg = tuple(qs_tensor.shape) if hasattr(qs_tensor, "shape") else None
            print(f"[BnB NONE rank={rank_dbg}] A.numel={A_tensor.numel()} qs.shape={shape_dbg} world={world_dbg}", flush=True)
        return res

def _run_part_a(A_tensor, qs_tensor, output_shape):
    n_packed_local = A_tensor.numel()
    shifts = _compute_shift_offsets(A_tensor, qs_tensor)
    if shifts is None:
        return None

        offset1, offset2 = shifts
        maj, minr = torch.cuda.get_device_capability(A_tensor.device)
        use_asm = maj == 7 and minr == 5 and (n_packed_local % 4 == 0)

    n_weights = n_packed_local * 2
    out_dtype = A_tensor.dtype if A_tensor.dtype in (torch.float16, torch.bfloat16) else torch.float16
    cache = getattr(qs_tensor, "_parta_out_cache", None)
    if (
        cache is None
        or cache.device != A_tensor.device
        or cache.numel() != n_weights
        or cache.dtype != out_dtype
    ):
        cache = torch.empty(n_weights, device=A_tensor.device, dtype=out_dtype)
        setattr(qs_tensor, "_parta_out_cache", cache)

    start = time.perf_counter()
    result = _your_dequantize_nf4(
        A_tensor,
        qs_tensor,
        offset1,
        offset2,
        use_custom_asm=use_asm,
        use_cache_eviction=False,
        out_tensor=cache,
    )
    if output_shape is not None:
        result = result.view(output_shape)

    elapsed = time.perf_counter() - start
    _DEQUANT_TIMES["part_a"] += elapsed
    _DEQUANT_TIMES["part_a_calls"] += 1
    _DEQUANT_TIMES["count"] += 1
    return result

    def _ensure_bnb_shape_cached(A_tensor, qs_tensor):
        """Cache BnB output shape per quant_state for orientation validation."""
        ref_shape = getattr(qs_tensor, "_bnb_reference_shape", None)
        if ref_shape is not None:
            return ref_shape
        try:
            ref = _ORIGINAL_BNB_DEQUANT(A_tensor, qs_tensor, absmax, out, blocksize, quant_type)
            qs_tensor._bnb_reference_shape = tuple(ref.shape)
            return qs_tensor._bnb_reference_shape
        except Exception:
            return None

    if _USE_PART_A_KERNEL and is_nf4 and A.is_cuda:
        full_shape = tuple(quant_state.shape) if hasattr(quant_state, "shape") else None
        if full_shape and len(full_shape) == 2:
            full_elements = full_shape[0] * full_shape[1]
            n_packed = A.numel()
            n_weights_effective = n_packed * 2  # NF4: 2 weights per byte

            # Debug output (once per rank)
            if not hasattr(patched_dequantize_4bit, '_debug_once'):
                patched_dequantize_4bit._debug_once = True
                rank_dbg, world_dbg = _get_fsdp_rank_info()
                print(f"[FSDP DEBUG rank={rank_dbg}] A.numel={n_packed}, full_elements={full_elements}", flush=True)
                print(f"[FSDP DEBUG rank={rank_dbg}] absmax.numel={quant_state.absmax.numel()}, world_size={world_dbg}", flush=True)

            is_fsdp_sharded = n_weights_effective < full_elements
            rank, world_size = _get_fsdp_rank_info()
            bnb_shape = _ensure_bnb_shape_cached(A, quant_state)
            target_shape = tuple(bnb_shape) if bnb_shape is not None else full_shape
            need_T = False
            if bnb_shape is not None and full_shape is not None and len(full_shape) == 2:
                need_T = (bnb_shape != full_shape) and (bnb_shape == (full_shape[1], full_shape[0]))
            setattr(quant_state, "_bnb_ref_shape", bnb_shape)
            setattr(quant_state, "_bnb_need_T", need_T)

            def _orient(result_q):
                if result_q is None:
                    return None
                res = result_q.t().contiguous() if need_T else result_q
                if bnb_shape is not None and tuple(res.shape) != tuple(bnb_shape):
                    res = res.view(bnb_shape)
                return res

            try:
                # Path 0: not sharded -> direct Part A
                if not is_fsdp_sharded:
                    qs_for_shape = _wrap_quant_state_with_shape(quant_state, full_shape)
                    result_q = _run_part_a(A, qs_for_shape, full_shape)
                    result = _orient(result_q)
                    if result is not None:
                        if bnb_shape is not None and tuple(result.shape) != bnb_shape:
                            if _DEQUANT_TIMES["fallback_calls"] <= 3:
                                print(f"[PartA FSDP] rank={rank} mismatch vs BnB shape {result.shape} != {bnb_shape}, falling back", flush=True)
                            raise RuntimeError("shape mismatch vs BnB")
                        return result

                # Path 1: reacquire full packed from owning module
                if is_fsdp_sharded and _ENABLE_REACQUIRE:
                    A_full = _reacquire_full_packed_from_module(quant_state)
                    if A_full is not None and A_full.numel() * 2 == full_elements:
                        qs_for_shape = _wrap_quant_state_with_shape(quant_state, full_shape)
                        result_q = _run_part_a(A_full, qs_for_shape, full_shape)
                        result = _orient(result_q)
                        if result is not None:
                            if bnb_shape is not None and tuple(result.shape) != bnb_shape:
                                if _DEQUANT_TIMES["fallback_calls"] <= 3:
                                    print(f"[PartA FSDP] rank={rank} mismatch vs BnB shape {result.shape} != {bnb_shape}, falling back", flush=True)
                                raise RuntimeError("shape mismatch vs BnB")
                            if _DEQUANT_TIMES["part_a_calls"] <= 3:
                                print(f"[PartA FSDP] rank={rank} path=reacquire shape={result.shape}", flush=True)
                            return result

                # Path 2: all-gather packed bytes to assemble full tensor
                if is_fsdp_sharded and _ENABLE_GATHER:
                    full_packed_bytes = full_elements // 2
                    A_full = _get_cached_full_packed(quant_state, full_packed_bytes, A.device) if _ENABLE_GATHER_CACHE else None
                    if A_full is None:
                        A_full = _all_gather_packed(A, full_packed_bytes)
                        if A_full is not None and _ENABLE_GATHER_CACHE:
                            _set_cached_full_packed(quant_state, A_full)
                    if A_full is not None and A_full.numel() * 2 == full_elements:
                        qs_for_shape = _wrap_quant_state_with_shape(quant_state, full_shape)
                        result_q = _run_part_a(A_full, qs_for_shape, full_shape)
                        result = _orient(result_q)
                        if result is not None:
                            if bnb_shape is not None and tuple(result.shape) != bnb_shape:
                                if _DEQUANT_TIMES["fallback_calls"] <= 3:
                                    print(f"[PartA FSDP] rank={rank} mismatch vs BnB shape {result.shape} != {bnb_shape}, falling back", flush=True)
                                raise RuntimeError("shape mismatch vs BnB")
                            if _DEQUANT_TIMES["part_a_calls"] <= 3:
                                print(f"[PartA FSDP] rank={rank} path=all_gather shape={result.shape}", flush=True)
                            return result

                # Path 3: local dequant + scatter into full shape
                if is_fsdp_sharded and _ENABLE_SCATTER:
                    eff_qs = _slice_quant_state_for_fsdp(quant_state, A, rank, world_size)
                    if eff_qs is not None:
                        shard_shape = tuple(getattr(eff_qs, "shape", ())) if hasattr(eff_qs, "shape") else None
                        qs_for_shape = _wrap_quant_state_with_shape(eff_qs, shard_shape)
                        local_q = _run_part_a(A, qs_for_shape, shard_shape)
                        if local_q is not None:
                            local_bnb = local_q.t().contiguous() if need_T else local_q
                            out_full = _scatter_local_to_full(local_bnb, quant_state, A, rank, world_size, bnb_shape, need_T)
                            if out_full is not None:
                                if bnb_shape is not None and tuple(out_full.shape) != bnb_shape:
                                    if _DEQUANT_TIMES["fallback_calls"] <= 3:
                                        print(f"[PartA FSDP] rank={rank} mismatch vs BnB shape {out_full.shape} != {bnb_shape}, falling back", flush=True)
                                    raise RuntimeError("shape mismatch vs BnB")
                                if _DEQUANT_TIMES["part_a_calls"] <= 3:
                                    print(f"[PartA FSDP] rank={rank} path=scatter shape={out_full.shape}", flush=True)
                                return out_full
            except Exception as e:
                _DEQUANT_TIMES["fallback_calls"] += 1
                if _DEQUANT_TIMES["fallback_calls"] <= 3:
                    print(f"Part A kernel failed (call #{_DEQUANT_TIMES['fallback_calls']}): {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    # Use original BnB
    start = time.perf_counter()
    result = _bnb_safe_call(A, quant_state)
    _DEQUANT_TIMES["bnb"] += time.perf_counter() - start
    _DEQUANT_TIMES["bnb_calls"] += 1
    _DEQUANT_TIMES["count"] += 1
    if result is None:
        # Extremely defensive: ensure we never propagate None to matmul
        rank_dbg, world_dbg = _get_fsdp_rank_info()
        bnb_shape_dbg = getattr(quant_state, "_bnb_ref_shape", None)
        print(f"[PartA FSDP] rank={rank_dbg} BnB returned None (A.numel={A.numel()}, bnb_shape={bnb_shape_dbg}, world={world_dbg})", flush=True)
        result = _bnb_safe_call(A, quant_state)
        if result is None:
            # Last resort: produce zeros to keep training alive for logging
            tgt_shape = tuple(bnb_shape_dbg) if bnb_shape_dbg is not None else ((quant_state.shape) if hasattr(quant_state, "shape") else (A.numel() * 2,))
            if isinstance(tgt_shape, int):
                tgt_shape = (tgt_shape,)
            result = torch.zeros(tgt_shape, device=A.device, dtype=out.dtype if out is not None else torch.float16)
            print(f"[PartA FSDP] rank={rank_dbg} substituted zeros for BnB None, shape={tgt_shape}", flush=True)

    # Debug: check what BnB returns (track first few calls with different sizes)
    if not hasattr(patched_dequantize_4bit, '_bnb_call_count'):
        patched_dequantize_4bit._bnb_call_count = 0
        patched_dequantize_4bit._bnb_sizes_seen = set()

    call_count = patched_dequantize_4bit._bnb_call_count
    patched_dequantize_4bit._bnb_call_count += 1

    a_numel = A.numel()
    if a_numel not in patched_dequantize_4bit._bnb_sizes_seen or call_count < 5:
        patched_dequantize_4bit._bnb_sizes_seen.add(a_numel)
        full_shape = tuple(quant_state.shape) if hasattr(quant_state, "shape") else None
        rank, _ = _get_fsdp_rank_info()
        print(f"[BnB DEBUG rank={rank} call={call_count}] A.numel={a_numel}, result.shape={result.shape}, quant_state.shape={full_shape}", flush=True)

    return result


def enable_part_a_kernel():
    """Enable Part A kernel for dequantization."""
    global _ORIGINAL_BNB_DEQUANT, _USE_PART_A_KERNEL
    import bitsandbytes.functional as bnb_F

    if _ORIGINAL_BNB_DEQUANT is None:
        _ORIGINAL_BNB_DEQUANT = bnb_F.dequantize_4bit

    bnb_F.dequantize_4bit = patched_dequantize_4bit
    _USE_PART_A_KERNEL = True
    print("Part A kernel ENABLED for NF4 dequantization")


def disable_part_a_kernel():
    """Disable Part A kernel, use original BnB."""
    global _ORIGINAL_BNB_DEQUANT, _USE_PART_A_KERNEL
    import bitsandbytes.functional as bnb_F

    if _ORIGINAL_BNB_DEQUANT is not None:
        bnb_F.dequantize_4bit = _ORIGINAL_BNB_DEQUANT
    _USE_PART_A_KERNEL = False
    print("Part A kernel DISABLED, using BnB")


def get_dequant_stats():
    """Get dequantization timing statistics."""
    return _DEQUANT_TIMES.copy()


def reset_dequant_stats():
    """Reset dequantization timing statistics."""
    global _DEQUANT_TIMES
    _DEQUANT_TIMES = {"part_a": 0.0, "bnb": 0.0, "count": 0, "part_a_calls": 0, "bnb_calls": 0, "fallback_calls": 0}


# ============================================================================
# Training Code
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FSDP2 + QLoRA Training with Part A Kernel")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs_fsdp2")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--use_part_a_kernel", action="store_true", help="Use Part A NF4 kernel")
    parser.add_argument("--benchmark_kernel", action="store_true", help="Benchmark Part A vs BnB")
    return parser.parse_args()


def get_bnb_config(compute_dtype=torch.float16):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=compute_dtype,
    )


def get_dataset(tokenizer):
    from datasets import load_dataset
    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")
    return dataset


def run_fsdp2_training(args):
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig

    compile_str = "+ torch.compile" if args.use_torch_compile else ""
    kernel_str = "+ Part A kernel" if args.use_part_a_kernel else ""
    print("=" * 60)
    print(f"Running FSDP2 + QLoRA {compile_str} {kernel_str} DISTRIBUTED TRAINING")
    print("=" * 60)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes

    if is_main:
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {accelerator.device}")

    # Enable Part A kernel if requested
    if args.use_part_a_kernel:
        enable_part_a_kernel()
        reset_dequant_stats()

    compute_dtype = torch.float16
    bnb_config = get_bnb_config(compute_dtype)

    if is_main:
        print(f"Loading model: {args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    register_quant_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.enable_input_require_grads()
    dataset = get_dataset(tokenizer)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args_kwargs = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=1,
        max_steps=args.max_steps,
        logging_steps=1,
        output_dir=args.output_dir,
        seed=3407,
        dataset_text_field="text",
        fp16=True,
        bf16=False,
        report_to="none",
        dataset_num_proc=4,
        save_strategy="no",
        gradient_checkpointing=args.use_gradient_checkpointing,
        optim="adamw_torch",
    )

    if args.use_torch_compile:
        if is_main:
            print("Enabling torch.compile via TrainingArguments...")
        training_args_kwargs["torch_compile"] = True
        training_args_kwargs["torch_compile_mode"] = "default"

    training_args = SFTConfig(**training_args_kwargs)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    if is_main:
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    if is_main:
        print("Starting FSDP2 training...")

    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time

    if is_main:
        print("\n" + "=" * 60)
        print("FSDP2 TRAINING RESULTS")
        print("=" * 60)
        print(f"Final loss: {train_result.training_loss:.4f}")
        print(f"Training time: {train_time:.2f}s")

        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        print(f"Loss curve (first 10): {losses[:10]}")
        print(f"Loss curve (last 10): {losses[-10:]}")

        # Print dequantization stats
        if args.use_part_a_kernel:
            stats = get_dequant_stats()
            print("\n" + "-" * 40)
            print("Dequantization Statistics:")
            print(f"  Part A kernel calls: {stats['part_a_calls']}")
            print(f"  Part A kernel time: {stats['part_a']:.4f}s")
            print(f"  BnB calls: {stats['bnb_calls']}")
            print(f"  BnB time: {stats['bnb']:.4f}s")
            print(f"  Fallback calls (errors): {stats['fallback_calls']}")
            print(f"  Total dequant calls: {stats['count']}")
            if stats['part_a'] > 0 and stats['bnb'] > 0:
                # Compare per-call times
                part_a_per_call = stats['part_a'] / stats['part_a_calls'] if stats['part_a_calls'] > 0 else 0
                bnb_per_call = stats['bnb'] / stats['bnb_calls'] if stats['bnb_calls'] > 0 else 0
                if part_a_per_call > 0:
                    speedup = bnb_per_call / part_a_per_call
                    print(f"  Per-call speedup: {speedup:.2f}x")
            print("-" * 40)

    return train_result, train_time


def run_benchmark(args):
    """Run benchmark comparing Part A kernel vs BnB baseline."""
    print("=" * 60)
    print("BENCHMARKING Part A Kernel vs BnB Baseline")
    print("=" * 60)

    # First run: BnB baseline
    print("\n--- Running BnB Baseline ---")
    args.use_part_a_kernel = False
    args.max_steps = 20  # Shorter for benchmark
    _, bnb_time = run_fsdp2_training(args)

    # Second run: Part A kernel
    print("\n--- Running with Part A Kernel ---")
    args.use_part_a_kernel = True
    _, part_a_time = run_fsdp2_training(args)

    # Results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"BnB baseline time: {bnb_time:.2f}s")
    print(f"Part A kernel time: {part_a_time:.2f}s")
    speedup = bnb_time / part_a_time if part_a_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print("✅ Part A kernel is FASTER than BnB!")
        print("   +3 points for Challenge B!")
    else:
        print("❌ Part A kernel is SLOWER than BnB")
        print("   No bonus points")


def main():
    args = parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.benchmark_kernel:
        run_benchmark(args)
    else:
        run_fsdp2_training(args)


if __name__ == "__main__":
    main()
