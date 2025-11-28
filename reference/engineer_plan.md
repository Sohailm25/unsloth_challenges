# Action Plan: NF4 to FP16/BF16 Single Triton Kernel

Let me break down this challenge systematically. This is a fascinating optimization problem that combines quantization theory with GPU kernel engineering.

## Understanding the Problem

### What is NF4 Quantization?

NF4 (4-bit NormalFloat) is a quantization scheme from QLoRA that:
1. **Assumes weights are normally distributed** - uses quantile-based mapping
2. **Uses block-wise quantization** - typically 64 elements per block
3. **Has double quantization** - the absmax values themselves are quantized to 8-bit

### The Double Dequantization Pipeline

```
NF4 (4-bit) ──────────────────────────────────────────► FP16/BF16
     │                                                      ▲
     │  Step 1: Lookup NF4 code → normalized value          │
     │  Step 2: Dequantize absmax (8-bit → FP32)            │
     │  Step 3: Scale normalized value by absmax            │
     └──────────────────────────────────────────────────────┘
```

The challenge is doing all three steps in **one kernel** without intermediate buffers.

---

## Phase 1: Deep Research & Analysis

### 1.1 Study Reference ImplementationsExcellent! Let me fetch the detailed blog post that achieved 1.6x-1.8x speedups:This is an excellent detailed reference. Now let me look at the uploaded file structure:Now I have a comprehensive understanding. Let me create a detailed action plan:

---

# Comprehensive Action Plan: NF4 to FP16/BF16 Single Triton Kernel

## Executive Summary

This challenge requires implementing a **fused Triton kernel** that performs double dequantization of NF4 quantized weights in a single kernel launch, outperforming Unsloth's `fast_dequantize` by ≥1.15x on Tesla T4.

---

## Phase 1: Deep Understanding (Day 1)

### 1.1 Data Structure Comprehension

The NF4 quantization scheme involves a hierarchical structure:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL QUANTIZATION STRUCTURE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 1: Weights (4-bit packed in uint8)                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ W[0:64]  │ W[64:128] │ W[128:192] │ ...  │ W[N-64:N]               │ │
│  │ blocksize=64 weights share same absmax                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│       │            │             │                                       │
│       ▼            ▼             ▼                                       │
│  Level 2: Absmax (quantized to uint8, fp8 codebook)                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ absmax[0] │ absmax[1] │ absmax[2] │ ...  │ absmax[N/64]            │ │
│  │ blocksize2=256 absmax values share same absmax2 (scale)            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│       │            │             │                                       │
│       ▼            ▼             ▼                                       │
│  Level 3: Absmax2 (fp32 scales) + Offset (mean)                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ absmax2[0] │ absmax2[1] │ ...  │ absmax2[(N/64)/256]               │ │
│  │ + global offset (mean of all absmax before 2nd quantization)       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Constants & Sizes

| Parameter | Value | Description |
|-----------|-------|-------------|
| `blocksize` | 64 | Weights per absmax block |
| `blocksize2` | 256 | Absmax values per absmax2 block |
| NF4 LUT | 16 floats | Lookup table for 4-bit → float |
| FP8 codebook | 256 floats | Lookup table for 8-bit absmax → float |

### 1.3 The NF4 Lookup Table (Hardcoded)

```python
NF4_CODES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 
    0.7229568362236023, 1.0
]
```

---

## Phase 2: Architecture Design (Day 1-2)

### 2.1 Dequantization Pipeline (What the Kernel Must Do)

```python
# STEP 1: For each packed weight byte W[i]:
#   - Extract high 4 bits: idx_h = W[i] >> 4
#   - Extract low 4 bits:  idx_l = W[i] & 0x0F

# STEP 2: Compute absmax index
#   - absmax_idx = (weight_position) // blocksize  (64)

# STEP 3: Double dequantize the absmax
#   - absmax_quant = absmax_quantized[absmax_idx]  (uint8)
#   - code_val = fp8_code[absmax_quant]            (float32)
#   - absmax2_idx = absmax_idx // blocksize2       (256)
#   - scale = absmax2[absmax2_idx]                 (float32)
#   - absmax = code_val * scale + offset           (float32)

# STEP 4: Final dequantization
#   - weight_h = NF4_LUT[idx_h] * absmax
#   - weight_l = NF4_LUT[idx_l] * absmax

# STEP 5: Store interleaved output
#   - out[2*i]   = weight_h
#   - out[2*i+1] = weight_l
```

### 2.2 Critical Optimization Insight: Memory Coalescing

The blog post revealed a **critical finding**: the naive interleaved store pattern destroys performance:

```
# BAD: Non-coalesced stores (causes 50% memory inefficiency)
tl.store(out_ptr + i*2, weight_h)      # stores to [0, 2, 4, 6, ...]
tl.store(out_ptr + i*2 + 1, weight_l)  # stores to [1, 3, 5, 7, ...]

# GOOD: Use tl.interleave + tl.reshape for coalesced writes
weights = tl.reshape(tl.interleave(weight_h, weight_l), 2 * BLOCK_SIZE)
tl.store(out_ptr + output_offset, weights)  # single contiguous write
```

### 2.3 Kernel Design V1 (Base)

```python
@triton.jit
def dequantize_nf4_fused_kernel(
    weight_ptr,       # Input: packed uint8 NF4 weights
    out_ptr,          # Output: fp16/bf16 dequantized weights
    absmax_ptr,       # Input: quantized absmax (uint8)
    absmax2_ptr,      # Input: absmax scale factors (fp32)
    code2_ptr,        # Input: FP8 codebook for absmax (256 fp32)
    offset,           # Input: scalar mean offset
    n_elements,       # Total packed weight elements (N/2)
    blocksize: tl.constexpr,       # 64
    blocksize2: tl.constexpr,      # 256
    BLOCK_SIZE: tl.constexpr,      # Tunable: 256, 512, 1024
):
    pid = tl.program_id(0)
    
    # Packed weight offsets (input is N/2 uint8 bytes)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # === DOUBLE DEQUANT OF ABSMAX ===
    # Each packed byte represents 2 weights at positions 2*offset and 2*offset+1
    # Both share the same absmax
    weight_positions = offsets * 2  # Actual weight positions (for absmax lookup)
    
    # Level 1: Get absmax index (blocksize=64)
    absmax_idx = weight_positions // blocksize
    
    # Level 2: Load quantized absmax and lookup in FP8 codebook
    absmax_quant = tl.load(absmax_ptr + absmax_idx, mask=mask)
    code_val = tl.load(code2_ptr + absmax_quant, mask=mask)
    
    # Level 3: Get absmax2 scale (blocksize2=256)
    absmax2_idx = absmax_idx // blocksize2
    scale = tl.load(absmax2_ptr + absmax2_idx, mask=mask)
    
    # Final absmax value
    absmax = code_val * scale + offset  # or use tl.fma for speed
    
    # === WEIGHT DEQUANTIZATION ===
    packed_weights = tl.load(weight_ptr + offsets, mask=mask)
    
    # Unpack: high nibble and low nibble
    idx_h = packed_weights >> 4
    idx_l = packed_weights & 0x0F
    
    # NF4 lookup (inline the table for best performance)
    weight_h = nf4_lookup(idx_h) * absmax
    weight_l = nf4_lookup(idx_l) * absmax
    
    # === COALESCED OUTPUT ===
    output = tl.reshape(tl.interleave(weight_h, weight_l), 2 * BLOCK_SIZE)
    output_offsets = pid * 2 * BLOCK_SIZE + tl.arange(0, 2 * BLOCK_SIZE)
    output_mask = output_offsets < (2 * n_elements)
    tl.store(out_ptr + output_offsets, output.to(out_ptr.dtype.element_ty), mask=output_mask)
```

### 2.4 NF4 Lookup Implementation Options

**Option A: tl.where cascade (branching)**
```python
@triton.jit
def nf4_lookup(idx):
    # Nested ternary/where operations
    result = tl.where(idx == 0, -1.0, 
             tl.where(idx == 1, -0.6961928009986877,
             # ... etc
             ))
    return result
```

**Option B: Load from constant memory (recommended)**
```python
# Pre-load NF4 codes to shared/constant memory
NF4_CODES_PTR = ...  # pointer to constant memory
code_val = tl.load(NF4_CODES_PTR + idx)
```

**Option C: Inline bitsandbytes-style LUT (from kernels.cu)**
```python
@triton.jit  
def dDequantizeNF4(val):
    # Direct translation from CUDA
    # Uses conditional branching similar to BnB implementation
    ...
```

---

## Phase 3: Implementation Strategy (Day 2-3)

### 3.1 Project Structure

```
nf4_triton_dequant/
├── modal_app.py              # Modal deployment for T4
├── kernels/
│   ├── __init__.py
│   ├── dequant_v1_base.py    # Basic fused kernel
│   ├── dequant_v2_coalesced.py  # With tl.interleave optimization
│   ├── dequant_v3_tuned.py   # Autotuned version
│   └── nf4_lut.py            # NF4 lookup implementations
├── benchmark/
│   ├── benchmark.py          # Benchmarking harness
│   └── test_correctness.py   # Numerical validation
└── utils/
    ├── quant_state.py        # Handle BnB quant_state
    └── reference.py          # Unsloth fast_dequantize wrapper
```

### 3.2 Modal Setup for T4

```python
import modal

app = modal.App("nf4-dequant-triton")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1.0",
    "triton>=2.1.0",
    "bitsandbytes",
    "transformers",
    "unsloth",  # for reference implementation
)

@app.function(
    gpu=modal.gpu.T4(),  # Tesla T4 as required
    image=image,
    timeout=600,
)
def run_benchmark(...):
    ...
```

### 3.3 Autotune Configuration

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def dequantize_nf4_fused_kernel(...):
    ...
```

---

## Phase 4: Optimization Strategies (Day 3-4)

### 4.1 Memory Access Optimizations

1. **Eviction Policy**: Use `eviction_policy='evict_last'` for weights since they're only used once

2. **Absmax Caching**: The same absmax value is shared by 64 weights. Consider:
   - Loading absmax once per block of 32/64 threads
   - Using shared memory for absmax caching (if beneficial)

3. **Coalesced Loads**: Ensure weight loads are contiguous

### 4.2 Compute Optimizations

1. **FMA Operations**: Use `tl.fma(code_val, scale, offset)` instead of `code_val * scale + offset`

2. **Type Casting**: Minimize casts; do computation in fp32, cast only at final store

3. **NF4 LUT in Registers**: Small enough (16 values) to keep in registers

### 4.3 T4-Specific Considerations

- **No native bf16**: T4 (sm_75) doesn't have native bf16 support
  - Triton will emulate bf16 operations
  - May need to use fp16 or fp32 compute, then cast to bf16
- **Shared Memory**: T4 has 48KB shared memory per SM
- **L2 Cache**: 6MB total - absmax/absmax2 should fit well

---

## Phase 5: Testing & Validation (Day 4-5)

### 5.1 Correctness Testing

```python
def test_dequantize_function(your_dequantize, reference_dequantize, quant_state):
    """Use the test function from the challenge"""
    output_yours = your_dequantize(weight, quant_state)
    output_ref = reference_dequantize(weight, quant_state)
    
    # Check numerical accuracy
    assert torch.allclose(output_yours, output_ref, rtol=0.01, atol=0.01)
```

### 5.2 Benchmark Configurations (From Challenge)

Test shapes resembling LLaMA weights:
```python
BENCHMARK_CONFIGS = [
    # (batch, seq_len, hidden_dim, intermediate_dim)
    # These translate to weight matrices of various sizes
    (1, 1, 4096, 11008),      # 7B style
    (1, 1, 5120, 13824),      # 13B style  
    (1, 1, 8192, 22016),      # 30B style
    (1, 1, 8192, 28672),      # 65B style
]
```

### 5.3 Performance Targets

| Metric | Target |
|--------|--------|
| Speedup vs `fast_dequantize` | ≥1.15x |
| Memory overhead | Minimal (no large intermediate buffers) |
| Single kernel launch | Required |

---

## Phase 6: Iteration Plan

### Iteration 1: Baseline Fused Kernel
- Implement basic fused kernel with naive stores
- Validate correctness
- Establish baseline performance

### Iteration 2: Memory Coalescing Fix
- Apply `tl.interleave` + `tl.reshape` pattern
- Expect significant speedup (this was the key insight from the blog)

### Iteration 3: Autotuning
- Add autotune decorator
- Test various BLOCK_SIZE and num_warps combinations
- Profile with `triton.testing.perf_report`

### Iteration 4: Fine-tuning
- Experiment with eviction policies
- Test different NF4 LUT implementations
- Profile with Nsight Compute if needed

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| T4 bf16 limitations | Use fp16 for output, or fp32 compute → bf16 cast |
| Pointer arithmetic bugs | Extensive unit testing with known inputs |
| Memory coalescing issues | Profile with Nsight Compute, apply interleave pattern |
| Not hitting 1.15x target | The blog achieved 1.6-1.8x, so headroom exists |

---

## Next Steps

1. **Set up Modal environment** with T4 GPU
2. **Implement reference wrapper** for `fast_dequantize`
3. **Build V1 kernel** and validate correctness
4. **Apply coalescing optimization** (V2)
5. **Autotune and benchmark** against target

Shall I start implementing any of these phases? I'd recommend beginning with the Modal setup and a skeleton kernel to validate the pipeline end-to-end.
