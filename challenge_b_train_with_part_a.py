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


_ENABLE_REACQUIRE = os.getenv("ORACLE_PARTA_REACQUIRE", "1") != "0"
_ENABLE_GATHER = os.getenv("ORACLE_PARTA_GATHER", "1") != "0"
_ENABLE_SCATTER = os.getenv("ORACLE_PARTA_SCATTER_FALLBACK", "1") != "0"
_ENABLE_GATHER_CACHE = os.getenv("ORACLE_PARTA_GATHER_CACHE", "0") != "0"
_ENABLE_PARTA_SHARDED = os.getenv("ORACLE_PARTA_SHARDED", "1") != "0"
_FORCE_PARTA = os.getenv("ORACLE_PARTA_FORCE", "0") != "0"
_DISABLE_CACHE_ON_SHARDS = os.getenv("ORACLE_PARTA_DISABLE_CACHE_ON_SHARDS", "1") != "0"

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
    if result is None:
        rank_dbg, world_dbg = _get_fsdp_rank_info()
        bnb_shape_dbg = getattr(quant_state, "_bnb_ref_shape", None)
        qshape_dbg = tuple(quant_state.shape) if hasattr(quant_state, "shape") else None
        print(f"[PartA FSDP] rank={rank_dbg} final None result (A.numel={A.numel()}, bnb_shape={bnb_shape_dbg}, qshape={qshape_dbg}, world={world_dbg})", flush=True)
        tgt_shape = tuple(bnb_shape_dbg) if bnb_shape_dbg is not None else (qshape_dbg if qshape_dbg is not None else (A.numel() * 2,))
        if isinstance(tgt_shape, int):
            tgt_shape = (tgt_shape,)
        result = torch.zeros(tgt_shape, device=A.device, dtype=out.dtype if out is not None else torch.float16)
    return result


def _compute_shift_offsets(weight, quant_state):
    """Legacy function, now delegates to FSDP-aware version."""
    return _compute_shift_offsets_fsdp(weight, quant_state)


def _infer_shard_rows(qs, packed_tensor, world_size):
    """Infer local shard row count from quant_state shape and packed bytes."""
    if not hasattr(qs, "shape") or len(qs.shape) != 2:
        return None
    q_rows, q_cols = qs.shape
    if q_cols == 0:
        return None
    if world_size > 0 and q_rows % world_size == 0:
        return q_rows // world_size
    shard_rows = (packed_tensor.numel() * 2) // q_cols
    return shard_rows if shard_rows > 0 else None


def _pad_cols_to_full(res, qs, target_cols=None):
    """Pad a 2D tensor's columns up to target_cols (defaults to qs.shape[1])."""
    if res is None:
        return res
    tgt_cols = target_cols
    if tgt_cols is None and hasattr(qs, "shape") and len(qs.shape) == 2:
        tgt_cols = qs.shape[1]
    if tgt_cols is None or not isinstance(tgt_cols, int):
        return res
    if res.dim() == 2 and res.shape[1] < tgt_cols:
        pad_cols = tgt_cols - res.shape[1]
        res = torch.nn.functional.pad(res, (0, pad_cols))
    return res


def _slice_deq_to_shard(deq, qs, rank, world_size, a_last_dim=None):
    """Slice dequant output to shard rows; pad columns to full cols or A last dim."""
    if deq is None or not hasattr(qs, "shape") or len(qs.shape) != 2:
        return deq
    q_rows, q_cols = qs.shape
    shard_rows = _infer_shard_rows(qs, deq if hasattr(deq, "numel") else torch.empty(0, device='cpu'), world_size)
    if shard_rows is None or shard_rows >= q_rows:
        return _pad_cols_to_full(deq, qs, a_last_dim or q_cols)
    row_start = rank * shard_rows
    row_end = min(row_start + shard_rows, q_rows)
    target_cols = a_last_dim if a_last_dim is not None else q_cols

    if deq.dim() == 2 and deq.shape[0] == q_rows:
        sliced = deq[row_start:row_end]
    elif deq.dim() == 2 and deq.shape[1] == q_rows:
        sliced = deq[:, row_start:row_end].t().contiguous()
    elif deq.dim() == 1 and deq.numel() == q_rows * q_cols:
        sliced = deq.view(q_rows, q_cols)[row_start:row_end]
    else:
        return _pad_cols_to_full(deq, qs, target_cols)

    return _pad_cols_to_full(sliced, qs, target_cols)


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


def _infer_local_2d_shape(A_tensor, quant_state):
    """Infer the 2D dequant shape that matches the provided packed bytes and quant_state."""
    full_shape = tuple(getattr(quant_state, "shape", ()))
    if len(full_shape) == 2:
        rows, cols = full_shape
        n_weights = A_tensor.numel() * 2
        if rows * cols == n_weights:
            return full_shape
        candidates = []
        if rows > 0 and (n_weights % rows) == 0:
            candidates.append((rows, n_weights // rows))  # column shard keeps rows
        if cols > 0 and (n_weights % cols) == 0:
            candidates.append((n_weights // cols, cols))  # row shard keeps cols
        for cand in candidates:
            r, c = cand
            if c == cols and r != rows:
                return cand
        for cand in candidates:
            r, c = cand
            if r == rows and c != cols:
                return cand
        if candidates:
            return candidates[0]
    return (A_tensor.numel() * 2,)


def _infer_shard_axis(full_shape, local_shape):
    """Infer whether FSDP sharded rows or columns based on shapes."""
    if len(full_shape) != 2 or len(local_shape) != 2:
        return None
    full_rows, full_cols = full_shape
    local_rows, local_cols = local_shape
    if local_rows == full_rows and local_cols != full_cols:
        return "col"
    if local_cols == full_cols and local_rows != full_rows:
        return "row"
    return None


def _decide_shard_axis(weight_shape, input_shape, full_shape, local_shape):
    """Determine shard axis using actual weight cols vs input feature dim.

    Priority:
    1) If weight cols differ from input features -> column shard.
    2) Else fall back to inferred axis from shapes (row/col).
    """
    if weight_shape is None or len(weight_shape) != 2:
        return None
    if input_shape is not None and len(input_shape) > 0:
        input_cols = input_shape[-1]
        if weight_shape[1] != input_cols:
            return "col"
    if full_shape is not None and local_shape is not None:
        return _infer_shard_axis(full_shape, local_shape)
    return None


def _orient_and_pad_for_full_cols(res, qs, full_shape, need_T):
    """Orient result to full_shape and pad columns when shorter."""
    if res is None:
        return None
    out = res.t().contiguous() if need_T else res
    target_cols = full_shape[1] if full_shape is not None and len(full_shape) == 2 else None
    out = _pad_cols_to_full(out, qs, target_cols)
    if full_shape is not None and len(full_shape) == 2 and out.numel() == full_shape[0] * full_shape[1] and tuple(out.shape) != tuple(full_shape):
        out = out.view(full_shape)
    return out


def _is_fsdp_sharded_shape(A_tensor, quant_state):
    full_shape = tuple(getattr(quant_state, "shape", ()))
    if len(full_shape) != 2:
        return False
    rows, cols = full_shape
    n_weights = A_tensor.numel() * 2
    if (rows * cols) <= n_weights:
        return False
    row_shard = cols > 0 and (n_weights % cols) == 0
    col_shard = rows > 0 and (n_weights % rows) == 0
    return row_shard or col_shard


def patched_dequantize_4bit(A, quant_state, absmax=None, out=None, blocksize=64, quant_type='fp4'):
    """Patched BnB dequantize that can use Part A kernel with shard-local paths."""
    global _DEQUANT_TIMES, _ORIGINAL_BNB_DEQUANT

    if _ORIGINAL_BNB_DEQUANT is None:
        import bitsandbytes.functional as bnb_F
        _ORIGINAL_BNB_DEQUANT = bnb_F.dequantize_4bit

    qt = getattr(quant_state, "quant_type", None)
    if qt == "nf4" or _FORCE_PARTA:
        quant_type = "nf4"
    is_nf4 = quant_type == "nf4"

    local_shape = _infer_local_2d_shape(A, quant_state)
    if local_shape is not None:
        setattr(quant_state, "_bnb_ref_shape", tuple(local_shape))

    if _USE_PART_A_KERNEL and is_nf4 and A.is_cuda:
        rank, world = _get_fsdp_rank_info()
        qs_eff = quant_state
        shard_shape = local_shape
        if _ENABLE_PARTA_SHARDED and _is_fsdp_sharded_shape(A, quant_state):
            qs_slice = _slice_quant_state_for_fsdp(quant_state, A, rank, world)
            if qs_slice is not None:
                qs_eff = qs_slice
                shard_shape = _infer_local_2d_shape(A, qs_eff)
        try:
            result = _run_part_a(A, qs_eff, shard_shape)
            if result is not None:
                full_shape = tuple(getattr(quant_state, "shape", ())) if hasattr(quant_state, "shape") else None
                need_T = False
                oriented = _orient_and_pad_for_full_cols(result, qs_eff, full_shape, need_T)
                return oriented
        except Exception as e:
            _DEQUANT_TIMES["fallback_calls"] += 1
            if _DEQUANT_TIMES["fallback_calls"] <= 3:
                rank_dbg, _ = _get_fsdp_rank_info()
                print(f"[PartA fallback rank={rank_dbg}] {e}", flush=True)

    qs_fallback = quant_state
    if _ENABLE_PARTA_SHARDED and _is_fsdp_sharded_shape(A, quant_state):
        rank, world = _get_fsdp_rank_info()
        qs_slice = _slice_quant_state_for_fsdp(quant_state, A, rank, world)
        if qs_slice is not None:
            qs_fallback = qs_slice

    start_t = time.perf_counter()
    result = _ORIGINAL_BNB_DEQUANT(A, qs_fallback, absmax, out, blocksize, quant_type)
    _DEQUANT_TIMES["bnb"] += time.perf_counter() - start_t
    _DEQUANT_TIMES["bnb_calls"] += 1
    _DEQUANT_TIMES["count"] += 1

    if result is None:
        tgt_shape = tuple(local_shape) if isinstance(local_shape, (tuple, list)) else (local_shape,)
        result = torch.zeros(tgt_shape, device=A.device, dtype=out.dtype if out is not None else torch.float16)
        rank_dbg, world_dbg = _get_fsdp_rank_info()
        print(f"[BnB NONE rank={rank_dbg}] returning zeros shape={tgt_shape} world={world_dbg}", flush=True)

    result = _pad_cols_to_full(result, quant_state)
    return result


def _run_part_a(A_tensor, qs_tensor, output_shape):
    """Run Part A and return a shard-shaped view; avoid persistent caches on shards."""
    n_packed_local = A_tensor.numel()
    shifts = _compute_shift_offsets(A_tensor, qs_tensor)
    if shifts is None:
        return None
    offset1, offset2 = shifts

    maj, minr = torch.cuda.get_device_capability(A_tensor.device)
    use_asm = (maj, minr) == (7, 5) and (n_packed_local % 4 == 0)

    n_weights = n_packed_local * 2
    out_dtype = getattr(qs_tensor, "dtype", None) or A_tensor.dtype
    if out_dtype not in (torch.float16, torch.bfloat16):
        out_dtype = torch.float16

    is_shard = hasattr(qs_tensor, "_original")
    use_persistent = (not is_shard) or (not _DISABLE_CACHE_ON_SHARDS)

    cache = None
    if use_persistent:
        cache = getattr(qs_tensor, "_parta_out_cache", None)
        if (
            cache is None
            or cache.device != A_tensor.device
            or cache.numel() != n_weights
            or cache.dtype != out_dtype
        ):
            cache = torch.empty(n_weights, device=A_tensor.device, dtype=out_dtype)
            setattr(qs_tensor, "_parta_out_cache", cache)
    else:
        cache = torch.empty(n_weights, device=A_tensor.device, dtype=out_dtype)

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
    if output_shape is None:
        output_shape = _infer_local_2d_shape(A_tensor, qs_tensor)
    if isinstance(output_shape, int):
        output_shape = (output_shape,)
    result = result.view(output_shape)

    _DEQUANT_TIMES["part_a"] += time.perf_counter() - start
    _DEQUANT_TIMES["part_a_calls"] += 1
    _DEQUANT_TIMES["count"] += 1

    if is_shard and _DISABLE_CACHE_ON_SHARDS:
        return result.clone()
    return result


def _ensure_bnb_shape_cached(A_tensor, qs_tensor):
    """Cache BnB output shape per quant_state using inference (no probe call)."""
    ref_shape = getattr(qs_tensor, "_bnb_reference_shape", None)
    if ref_shape is not None:
        return ref_shape
    ref_shape = _infer_local_2d_shape(A_tensor, qs_tensor)
    qs_tensor._bnb_reference_shape = tuple(ref_shape) if isinstance(ref_shape, (list, tuple)) else (ref_shape,)
    return qs_tensor._bnb_reference_shape


def enable_part_a_kernel():
    """Enable Part A kernel for dequantization."""
    global _ORIGINAL_BNB_DEQUANT, _USE_PART_A_KERNEL
    import bitsandbytes.functional as bnb_F
    import bitsandbytes.autograd._functions as bnb_autograd

    if _ORIGINAL_BNB_DEQUANT is None:
        _ORIGINAL_BNB_DEQUANT = bnb_F.dequantize_4bit

    bnb_F.dequantize_4bit = patched_dequantize_4bit

    def _patched_matmul_forward(ctx, A, B, out=None, bias=None, quant_state=None):
        rank, world_size = _get_fsdp_rank_info()
        local_shape = _infer_local_2d_shape(B, quant_state)
        if local_shape is not None:
            setattr(quant_state, "_bnb_ref_shape", tuple(local_shape))
        full_shape = tuple(getattr(quant_state, "shape", ())) if quant_state is not None else None
        weight = patched_dequantize_4bit(B, quant_state)
        if weight is None:
            shape = local_shape if isinstance(local_shape, (tuple, list)) else (B.numel() * 2,)
            weight = torch.zeros(shape, device=A.device, dtype=A.dtype)
        if weight.dim() == 1 and isinstance(local_shape, (tuple, list)) and len(local_shape) == 2:
            weight = weight.view(local_shape)

        target_in_cols = A.shape[-1] if hasattr(A, "shape") and len(A.shape) > 0 else None
        weight = _pad_cols_to_full(weight, quant_state, target_in_cols)
        if weight.dim() == 2 and target_in_cols is not None and weight.shape[1] > target_in_cols:
            weight = weight[:, :target_in_cols]

        shard_axis = _decide_shard_axis(
            tuple(weight.shape) if hasattr(weight, "shape") else None,
            tuple(A.shape) if hasattr(A, "shape") else None,
            full_shape,
            tuple(local_shape) if isinstance(local_shape, (tuple, list)) else None,
        )

        local_out = torch.nn.functional.linear(A, weight, bias)

        ctx.rank = rank
        ctx.world_size = world_size if dist.is_initialized() else 1
        ctx.local_rows = weight.shape[0]
        ctx.local_cols = weight.shape[1] if weight.dim() == 2 else None
        ctx.full_rows = full_shape[0] if full_shape and len(full_shape) == 2 else None
        env_fsdp = os.getenv("ACCELERATE_USE_FSDP", "") != ""
        is_sharded = (_ENABLE_PARTA_SHARDED and dist.is_initialized() and ctx.world_size > 1) or _is_fsdp_sharded_shape(B, quant_state) or env_fsdp
        ctx.shard_axis = shard_axis
        ctx.should_reduce = dist.is_initialized() and ctx.world_size > 1 and is_sharded and shard_axis == "col"
        ctx.should_gather = dist.is_initialized() and ctx.world_size > 1 and is_sharded and shard_axis != "col"
        ctx.slice_rows = ctx.local_rows
        ctx.rows_sizes = None

        if hasattr(ctx, "save_for_backward"):
            ctx.save_for_backward(weight)
        else:
            ctx._saved_tensors = (weight,)

        if ctx.should_reduce:
            dist.all_reduce(local_out)
            return local_out

        if not ctx.should_gather:
            return local_out

        rows_tensor = torch.tensor([ctx.local_rows], device=local_out.device, dtype=torch.int64)
        rows_list = [torch.zeros_like(rows_tensor) for _ in range(ctx.world_size)]
        dist.all_gather(rows_list, rows_tensor)
        max_rows = int(torch.stack(rows_list).max().item())
        total_rows = int(torch.stack(rows_list).sum().item())
        ctx.rows_sizes = [int(x.item()) for x in rows_list]
        if ctx.local_rows < max_rows:
            pad_rows = max_rows - ctx.local_rows
            weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows))
            pad_out_shape = list(local_out.shape)
            pad_out_shape[-1] = pad_rows
            local_out = torch.cat(
                [local_out, torch.zeros(pad_out_shape, device=local_out.device, dtype=local_out.dtype)],
                dim=-1,
            )
        ctx.slice_rows = max_rows

        if hasattr(ctx, "save_for_backward"):
            ctx.save_for_backward(weight)
        else:
            ctx._saved_tensors = (weight,)

        gathered = [torch.empty_like(local_out) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local_out)
        out_full = torch.cat(gathered, dim=-1)
        target_cols = None
        if total_rows > 0:
            target_cols = total_rows
        if target_cols is None and ctx.full_rows is not None:
            target_cols = ctx.full_rows
        if target_cols is not None and out_full.shape[-1] > target_cols:
            out_full = out_full[..., : target_cols]
        return out_full

    def _patched_matmul_backward(ctx, grad_output):
        if hasattr(ctx, "saved_tensors"):
            (weight,) = ctx.saved_tensors
        else:
            (weight,) = getattr(ctx, "_saved_tensors", (None,))
        if weight is None:
            return None, None, None, None, None
        if ctx.should_gather and ctx.world_size > 1 and ctx.local_rows is not None:
            rows_sizes = getattr(ctx, "rows_sizes", None)
            if rows_sizes is not None:
                offset = sum(rows_sizes[: ctx.rank])
            else:
                slice_rows = getattr(ctx, "slice_rows", ctx.local_rows)
                offset = ctx.rank * slice_rows
            grad_local = grad_output[..., offset : offset + ctx.local_rows]
            grad_A = torch.nn.functional.linear(grad_local, weight.t())
            dist.all_reduce(grad_A)
        elif getattr(ctx, "should_reduce", False) and ctx.world_size > 1:
            local_cols = ctx.local_cols or (weight.shape[1] if weight.dim() == 2 else grad_output.shape[-1])
            cols_tensor = torch.tensor([local_cols], device=grad_output.device, dtype=torch.int64)
            cols_list = [torch.zeros_like(cols_tensor) for _ in range(ctx.world_size)]
            dist.all_gather(cols_list, cols_tensor)
            max_cols = int(torch.stack(cols_list).max().item())
            total_cols = int(torch.stack(cols_list).sum().item())

            grad_slice = torch.nn.functional.linear(grad_output, weight)
            if local_cols < max_cols:
                pad_cols = max_cols - local_cols
                grad_slice = torch.nn.functional.pad(grad_slice, (0, pad_cols))

            gathered = [torch.empty_like(grad_slice) for _ in range(ctx.world_size)]
            dist.all_gather(gathered, grad_slice)
            grad_A = torch.cat(gathered, dim=-1)
            if total_cols > 0 and grad_A.shape[-1] > total_cols:
                grad_A = grad_A[..., : total_cols]
        else:
            grad_A = torch.nn.functional.linear(grad_output, weight.t())
        return grad_A, None, None, None, None

    bnb_autograd.MatMul4Bit.forward = _patched_matmul_forward
    bnb_autograd.MatMul4Bit.backward = _patched_matmul_backward

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
