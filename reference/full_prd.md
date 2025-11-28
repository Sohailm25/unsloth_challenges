
Here’s a full PRD that folds in the experienced engineer’s plan, the public puzzle spec, and the rubric — and tightens everything up so you can drive to 14/14.

I’ll write it as if we’re starting from zero code.

---

# PRD: NF4 → FP16/BF16 Fused Triton Dequantization Kernel

**Owner:**
**Target hardware:** NVIDIA Tesla T4 (sm_75, 16GB)
**Target environment:** PyTorch + Triton, executed via Modal GPU jobs
**Goal:** Achieve the full 14/14 rubric score for the NF4 → FP16/BF16 single Triton kernel challenge.

---

## 1. Problem Statement

NF4 (NormalFloat4) NF4 weights are stored as:

* A packed `torch.uint8` tensor of weights (2 NF4 values per byte).
* A nested `QuantState` (bitsandbytes-style) holding blockwise absmax statistics and codebooks, including a double quantization of the absmax values themselves. ([Hugging Face][1])

Existing implementations dequantize in **two separate GPU kernels** (e.g. `cdequantize_blockwise_fp32` then `cdequantize_blockwise_fp16_nf4` or via Unsloth’s `fast_dequantize` wrapper). ([Medium][2])

The challenge:

> Implement **one** Triton kernel that:
>
> * Performs **both levels** of dequantization (absmax double-quant + NF4 weight dequant).
> * Outputs FP16 or BF16.
> * Is **≥ 1.15× faster** than Unsloth’s `fast_dequantize` on a Tesla T4.
> * Meets the rubric items (single kernel, custom asm, cache eviction, fp16+bf16, torch.compile-safe).

This PRD defines *what* must be delivered (scope & requirements) and the acceptance criteria; it does **not** prescribe a single “only way” to implement things.

---

## 2. Scope

### 2.1 In-Scope

1. **Primary kernel**: a fused Triton kernel that:

   * Accepts NF4-packed weights and a nested quantization state.
   * Performs **double dequantization** in one kernel:

     * Dequant absmax (nested 8‑bit structure).
     * Dequant NF4 packed weights using absmax.
   * Writes FP16 or BF16 weights to an output tensor.

2. **Python-level API** (host-side wrapper) that:

   * Integrates with an Unsloth-style `weight_obj` or similar wrapper (`weight.data`, `weight.quant_state`, `weight.data_shape`). ([Medium][3])
   * Exposes a stable function `your_dequantize_nf4(...)` (name TBD is fine, but must be fixed in this project).
   * Supports both FP16 and BF16 compute/outputs.

3. **Performance harness & reference implementation**:

   * Integration with Unsloth’s `fast_dequantize` via a wrapper.
   * Benchmark script to compare speed and memory.
   * Use of the given `test_dequantize_function` (from the puzzle) as the core correctness and performance harness.

4. **Modal integration**:

   * Modal app/function(s) to run correctness tests and benchmarks on a Tesla T4.
   * Simple CLI / entrypoints.

5. **Torch.compile compatibility testing**:

   * A small, self-contained test that wraps a module using the new dequant path inside `torch.compile(...)` and verifies:

     * Compilation succeeds.
     * Numerical equivalence to a non-compiled reference.

6. **Custom ASM variant & cache-eviction variant**:

   * A Triton kernel variant that uses inline/custom assembly via Triton’s `tl.asm` (or equivalent) and passes all tests on T4. ([Medium][3])
   * Optional eviction-buffer–based cache management, toggled at runtime.

### 2.2 Out-of-Scope

* Implementing or altering **quantization** (NF4 encoding, training, or bitsandbytes quantization routines).
* Supporting non-NF4 quantization types (fp4/af4/int4) beyond what is needed to parse and use the existing NF4 quant_state.
* New CUDA C++/C kernels (outside of Triton’s internal codegen).
* Non-T4 GPU-specific tuning (Ampere, Hopper, AMD, Intel). Kernel may work elsewhere, but T4 is the only required target.
* Full Unsloth / Transformers integration beyond what `test_dequantize_function` needs.

---

## 3. Users & Use Cases

### 3.1 Users

* **Challenge evaluators / puzzle runners**: call the provided test harness to evaluate correctness and performance.
* **Unsloth / NF4 researchers**: could drop this in as a replacement backend for NF4 dequant during QLoRA fine‑tuning. ([CSDN Blog][4])

### 3.2 Primary Use Case

* Dequantize NF4‑quantized LoRA / MLP weights within a forward pass:

  1. `weight_obj` has `.data` (torch.uint8), `.quant_state` (nested QuantState-like object), `.data_shape`, `.quant_state.dtype`.
  2. Call `your_dequantize_nf4(weight_obj, ...)`.
  3. Receive a dense FP16/BF16 weight matrix (same shape as original float weights) on GPU.
  4. Immediately feed into GEMM/linear operations.

---

## 4. Background & Data Model

### 4.1 NF4 Encoding & Packing

* NF4 uses a 16-level **non-uniform LUT** tuned for normally distributed weights; each 4-bit symbol maps to one float in [-1, 1]. ([Hugging Face][1])
* Two 4‑bit NF4 codes are packed into a single `uint8`:

  * Lower nibble: `byte & 0x0F`
  * Upper nibble: `byte >> 4`

So if the original weight tensor has `N_float` elements, then:

* Packed NF4 tensor size: `N_packed = N_float // 2` (`torch.uint8`).

### 4.2 Double Quantization / QuantState Layout

From bitsandbytes NF4 nested quantization: ([Hugging Face][1])

* At the Python level, NF4 quantization returns:

  * `data`: `torch.uint8` weights (`N_packed`).
  * `quant_state`: a `QuantState` represented as nested state:

    * `quant_state.absmax`: first-level absmax values, quantized (e.g. `torch.uint8`) in blocks of `blocksize` (typical: 64).
    * `quant_state.code`: first-level codebook for NF4 (not directly used in dequantizing nested absmax).
    * `quant_state.blocksize`: block size of the first-level quantization.
    * `quant_state.offset`: scalar float (mean offset of absmax before second quantization).
    * `quant_state.state2`: nested `QuantState` describing second-level quantization of `quant_state.absmax`:

      * `state2.absmax`: second-level absmax values.
      * `state2.code`: FP8-like codebook (length 256).
      * `state2.blocksize`: block size for second-level absmax (`blocksize2`).
      * `state2.dtype` etc.

Effective semantics (simplified target behavior):

1. **Dequantize absmax values**:

   * `absmax_quant` is `quant_state.absmax` (uint8 codes, one per block of weights).
   * `code2 = quant_state.state2.code` is the FP8-like 256-element codebook.
   * `absmax2 = quant_state.state2.absmax` is a per-block scaling factor for absmax codes.
   * The scalar `quant_state.offset` is added back.

2. **Reconstruct per-block absmax**:

   * For a block index `b`:

     * `code_val = code2[absmax_quant[b]]`
     * `scale = absmax2[b // (blocksize2/blocksize)]` (exact mapping defined by the block sizes).
     * `absmax[b] = code_val * scale + offset`  (often implemented with FMA in kernels).

3. **Dequantize NF4 weights**:

   * Each NF4 symbol index `i` maps to `NF4_LUT[i]` (16-element LUT).
   * Final float value: `float_val = NF4_LUT[nibble] * absmax[block_of_that_weight]`.

The fused Triton kernel must implement exactly this semantic mapping.

### 4.3 Block Size Invariants

* First-level block size `blocksize` is typically 64. ([Medium][2])
* Second-level block size `blocksize2` is typically 256 (absmax values grouped for the nested quantization). ([Medium][2])

**Requirement (R-bg-1):**
Host code must compute:

* `blocksize_from_state = quant_state.blocksize`
* `blocksize2_from_state = quant_state.state2.blocksize`

and pass their **log2** ratios as kernel `offset1`/`offset2` only if they are powers of two. The kernel or wrapper must validate that:

* `((N_packed * 2) // blocksize_from_state) == quant_state.absmax.numel()`
* `blocksize_from_state > 0`, `blocksize2_from_state > 0`
* If using bitshifts:

  * `(N_packed * 2 // quant_state.absmax.numel())` is a power of 2.
  * `(quant_state.absmax.numel() // quant_state.state2.absmax.numel())` is a power of 2.

If these invariants fail, the wrapper should **fall back** to a slower, correctness-first path (e.g. a non-bitshift implementation or reference dequant).

---

## 5. Functional Requirements

### 5.1 Core Dequantization API

**FR-1: Dequantization front-end function**

* Provide a function (final name TBD, but consistent within repo):

  ```python
  def your_dequantize_nf4(
      weight_obj,
      *,
      use_custom_asm: bool = False,
      use_cache_eviction: bool = False,
      use_optimized: bool = True,
  ) -> torch.Tensor:
      ...
  ```

* `weight_obj` contract (minimum interface):

  * `weight_obj.data`: `torch.uint8` tensor on CUDA, shape `(N_packed,)` or `(rows, cols_packed)`; contiguous.
  * `weight_obj.quant_state`: object with attributes:

    * `.absmax`, `.code`, `.blocksize`, `.offset`, `.state2` (with `.absmax`, `.code`, `.blocksize`, `.dtype`).
    * `.dtype`: output compute dtype (`torch.float16` or `torch.bfloat16`).
    * `.shape`: original float weight shape (e.g. `(out_features, in_features)`).
  * `weight_obj.data_shape` (optional): shape of output float weights. If not provided, use `quant_state.shape`.

* Behavior:

  1. Flattens `weight_obj.data` to 1D view (`N_packed`).
  2. Computes kernel parameters: `n_elements = N_packed`, `offset1`, `offset2`, grid dims, etc.
  3. Allocates `out_flat` of shape `(N_packed * 2,)`, dtype `torch.float16` (internal compute dtype).
  4. Calls `_your_dequantize_nf4` backend (see FR‑2).
  5. Reshapes `out_flat` to `weight_obj.data_shape` or `quant_state.shape`.
  6. Casts to `quant_state.dtype` if needed.
  7. Returns the reshaped, dequantized tensor.

**Acceptance criteria:**

* Works with both FP16 and BF16 target types.
* No `torch.compile` used inside this function.

---

**FR-2: Single-kernel fused Triton implementation**

Implement a Triton kernel (name & exact signature flexible) equivalent to:

```python
@triton.jit
def dequantize_nf4_fused_kernel(
    weight_ptr,      # uint8*
    out_ptr,         # fp16/bf16*
    absmax_ptr,      # uint8* or fp32*, depending on scheme
    absmax2_ptr,     # fp32*
    code2_ptr,       # fp32* (size 256)
    offset,          # fp32 scalar
    n_packed,        # int: number of packed bytes
    offset1,         # int: log2 ratio for absmax indexing
    offset2,         # int: log2 ratio for absmax2 indexing
    BLOCK_SIZE: tl.constexpr,
    # Additional pointers for cache eviction / asm variants if needed
):
    ...
```

Functional semantics per thread:

1. Compute a block of `offsets = pid * BLOCK_SIZE + arange(0, BLOCK_SIZE)` with `mask = offsets < n_packed`.
2. For each `offset`, compute the **absolute weight index** (`weight_pos = offsets * 2`) and block indices.
3. Load quantized absmax and reconstruct real absmax using nested quantization (`absmax`, `absmax2`, `code2`, `offset`).
4. Load packed NF4 byte.
5. Extract high and low nibbles and map them via NF4 LUT.
6. Multiply LUT values by the absmax for that block.
7. Interleave and store **two FP16/BF16 outputs per input byte** into `out_ptr` contiguously.

**Single-kernel requirement (Rubric +3 points):**

* All of steps (2–6) **must occur in this single kernel**.
* There must be **no second Triton kernel** for dequantization.
* No calls to bitsandbytes CUDA functions in the dequant path.

Verifier strategy:

* A small profiling/test helper that:

  * Runs the dequant path once under torch.profiler and confirms only one Triton kernel with our name is launched (excluding trivial type-cast kernels, memory sets, etc.).

---

### 5.2 Dtype Support

**FR-3: FP16 and BF16 outputs**

* The implementation must support:

  * `quant_state.dtype == torch.float16`
  * `quant_state.dtype == torch.bfloat16`

Behavior:

* Compute internally in FP32 inside the kernel for accuracy.
* Cast to the target dtype **at store-time** or in a post-step, but:

  * There must be **no full-size FP32 intermediate tensor** on the host (see NFR-3).

Acceptance:

* `test_dequantize_function` (or equivalent) must:

  * Run a set of FP16 cases.
  * Run a set of BF16 cases.
  * Verify outputs vs reference (`fast_dequantize`) to within given tolerances (see NFR-2 and FR-5). ([Medium][3])

---

### 5.3 Torch.compile Compatibility

**FR-4: torch.compile-safe kernel**

Even though we **must not use** `torch.compile` in our implementation, the kernel must be callable from code that is compiled, meeting the rubric:

> `if kernel_works_in_torch_compile: A_score += 1, else: A_score -= 1`

Requirements:

* Implement a test module:

  ```python
  class DequantLinear(nn.Module):
      def __init__(self, quant_linear_like):
          super().__init__()
          self.weight = quant_linear_like.weight  # carries .data, .quant_state, etc.

      def forward(self, x):
          W = your_dequantize_nf4(self.weight)
          return x @ W.t()
  ```

* Test:

  ```python
  compiled = torch.compile(DequantLinear(q_lin).cuda())
  out1 = compiled(x)
  out2 = DequantLinear(q_lin).cuda()(x)
  assert torch.allclose(out1, out2, atol=1e-1, rtol=1e-1)
  ```

Constraints:

* No Python-side dynamic control-flow inside the kernel launch path that confuses `torch.compile` (e.g. avoid depending on non-tensor global state).
* Any runtime decisions (pick asm vs non-asm, cache eviction etc.) made **before** compilation, or in a way that still yields a static graph.

---

### 5.4 Custom ASM Variant

**FR-5: Working custom_asm kernel path**

Rubric asks:

> `if custom_asm_works: A_score += 3`

Requirements:

* Provide one or more Triton kernels that use `tl.asm` (or equivalent Triton inline assembly) to speed up a hot part of the dequant computation, for example:

  * Vectorized loads (e.g. 4×uint8 at once) and byte extraction.
  * Efficient FP16/BF16 conversions or fused multiply-add sequences tailored to T4.

API:

* `your_dequantize_nf4(..., use_custom_asm=True, use_optimized=...)` must:

  * Dispatch to an asm-backed Triton kernel on compatible devices (T4).
  * Fall back to the vectorized non-asm kernel otherwise.

Acceptance:

* ASM variant passes **all** correctness tests and reliability checks on Tesla T4.
* ASM variant is involved in the benchmark used for speedup calculation (see NFR‑1), or at least does not regress performance.

---

### 5.5 Cache Eviction Strategy

**FR-6: Optional cache eviction support**

Rubric:

> `if uses_cache_eviction: A_score += 1` ([Medium][3])

Requirements:

* Implement an **optional** eviction-buffer–based cache strategy, similar to:

  * Host allocates an `evict` tensor (e.g. uint8 with length `BLOCK_SIZE`) if `use_cache_eviction=True`.
  * Kernel takes an extra `evict_ptr` argument.
  * Kernel performs a dummy load:

    ```python
    _ = tl.load(evict_ptr + offsets, mask=mask, other=0)
    ```

    before critical quantization parameter loads.

* `your_dequantize_nf4(..., use_cache_eviction=True)`:

  * Must activate this behavior.
  * Must still pass correctness tests.

Acceptance:

* There is at least one benchmark run with `use_cache_eviction=True` (even if the speed difference is small).
* No correctness regressions when enabling cache eviction.

---

### 5.6 Correctness vs Reference

**FR-7: Functional equivalence to Unsloth/bitsandbytes**

Using the provided `test_dequantize_function`, or an equivalent that matches the puzzle spec:

* For each test configuration:

  * NF4 quantized weights are generated using bitsandbytes/Unsloth’s quantization pipeline.
  * Reference dequantization uses `unsloth.kernels.utils.fast_dequantize` (or its direct C backend). ([CSDN Blog][4])
  * The new Triton-based implementation must produce outputs that are **numerically close**:

    * `torch.allclose(new, ref, rtol=0.01, atol=0.01)` for element-wise weight comparison **or**
    * `torch.allclose(mlp_output_new, mlp_output_ref, atol=1e-1)` for end-to-end MLP outputs, as used in the blog/harness. ([Medium][3])

* Tests must cover:

  * FP16 and BF16 compute dtypes.
  * Multiple shapes reflecting LLaMA-style MLP matrices (see NFR‑1).

If the puzzle’s `test_dequantize_function` includes stricter or slightly different thresholds, those thresholds always take precedence.

---

## 6. Non-Functional Requirements

### 6.1 Performance

**NFR-1: Speedup vs Unsloth fast_dequantize**

Rubric requires ≥1.15× speedup to get full score increments. We need to formalize *how* speedup is calculated.

**Baseline:**

* Use Unsloth’s published `fast_dequantize` on the same hardware and environment as reference. ([CSDN Blog][4])
* Pin a specific Unsloth version (log its version + commit hash in benchmark output).

**Workloads:**

Use the same or very similar workloads as described in the puzzle/blog:

* Several (batch size, seq length, hidden dim, MLP dim, seed, dtype) tuples, e.g. (example set from article): ([Medium][3])

  * Case A (FP16): e.g. `(bsz=2, qlen=3333, hd=2048, m=8192, seed, torch.float16)`
  * Case B (BF16): e.g. `(bsz=5, qlen=777, hd=1024, m=4096, seed, torch.bfloat16)`
  * Case C (BF16): e.g. `(bsz=3, qlen=2048, hd=4096, m=14336, seed, torch.bfloat16)`
  * Plus additional LLaMA-style settings (7B/13B/30B/65B MLP dimensions) as needed.

**Measurement methodology:**

* Warm up both reference and new kernel sufficiently (e.g., 5–10 iterations).
* Use `torch.cuda.Event` timing and multiple runs per configuration (e.g. 50–100) and take median or mean.
* Repeat entire experiment at least 3 times and report average speedup.

**Definition of speedup:**

* For each configuration: `speedup_i = T_ref_i / T_new_i`.
* Report:

  * `min_i speedup_i`
  * `mean_i speedup_i`.

**Acceptance threshold:**

* Hard requirement:

  * `min_i speedup_i >= 1.05` (no case slower than reference by more than measurement noise).
* Rubric target:

  * `mean_i speedup_i >= 1.15` (to credibly claim ≥1.15× speedup overall).
* If `mean` is slightly under 1.15 but `min` and `median` are high, treat as a risk / open issue; still record results.

---

### 6.2 Numerical Accuracy

**NFR-2: Accuracy tolerance**

* For direct weight comparison:

  * `rtol ≤ 0.01`, `atol ≤ 0.01` (or stricter, depending on test harness).
* For end-to-end MLP output (forward):

  * `atol ≤ 1e-1` as seen in the published challenge harness. ([Medium][3])

These constraints apply across both FP16 and BF16 tests.

---

### 6.3 Memory & Buffers

**NFR-3: No large intermediate buffers**

Constraints:

* In the **host code**:

  * No full-size FP32 dequantization buffers in addition to the final FP16/BF16 output tensor.
  * Temporary tensors should not exceed 10% of the size of the final dequantized tensor, aside from the final output itself (rule-of-thumb; exact ratio can be validated via peak memory measurement).

* In the **kernel**:

  * Intermediate arrays (like `deq_h`, `deq_l`, or an interleaved buffer) must be per-block, residing in registers / local memory.
  * No global scratch tensor with full weight size.

**Detection:**

* Compare peak CUDA memory usage between:

  * Reference implementation.
  * New implementation with large weights.
* Ensure no large one-shot allocations (other than the result tensor and the existing quantization state).

---

### 6.4 T4 Hardware Constraints

**NFR-4: T4 compatibility**

* Kernel must compile and run correctly on sm_75.
* No reliance on hardware features absent on T4 (e.g., TensorCore BF16 ops are not available natively).
* BF16 support may be emulated via FP32 compute + cast; this is acceptable as long as performance targets are met.

Testing:

* All correctness and performance benchmarks run on a Tesla T4 via Modal (see Section 7).

---

## 7. Deployment & Modal Integration

**FR-8: Modal app for GPU runs**

Requirements:

* Provide a `modal.App` (e.g. `nf4-dequant-triton`) with:

  * An image that installs:

    * `torch` (>= 2.1 or puzzle-specified version).
    * `triton`.
    * `bitsandbytes`.
    * `unsloth`.
  * GPU type `modal.gpu.T4()`.

* Expose at least one public function:

  ```python
  @app.function(gpu=modal.gpu.T4(), ...)
  def run_benchmarks(...):
      # runs test_dequantize_function, prints/returns timing and correctness stats
  ```

* CLI / usage:

  * `modal run modal_app.py::run_benchmarks` (or equivalent) should:

    * Run all correctness tests (FP16 + BF16).
    * Run benchmarks vs `fast_dequantize`.
    * Print summary and optionally save to JSON (for reproducibility).

---

## 8. API & Integration Details

### 8.1 Python API Surface

Minimal set:

* `your_dequantize_nf4(weight_obj, use_custom_asm=False, use_cache_eviction=False, use_optimized=True)`
* Internal backend: `_your_dequantize_nf4(data, quant_state, use_custom_asm, use_cache_eviction, use_optimized)` returning a flat dequantized tensor.
* Kernel exposure:

  * `dequantize_nf4_fused_kernel` (vectorized main kernel).
  * `dequantize_nf4_fused_kernel_asm` (optional asm variant).

### 8.2 NF4 LUT Implementation

Options allowed (implementation choice):

* Hard-coded NF4 LUT as constants inside the kernel.
* Or loading from a small constant buffer (`tl.load` from a static 16-element LUT) created on host.

Constraints:

* LUT values must match bitsandbytes NF4 levels (normalized quantiles). ([Hugging Face][1])
* Implementation detail (branching vs table lookup) is up to the design, but it must not materially harm performance.

---

## 9. Testing & Validation

### 9.1 Provided Puzzle Test: `test_dequantize_function`

Requirements:

* Integrate the official `test_dequantize_function` exactly as given in the puzzle (once you have the file in your repo).

* Add wrappers so that:

  * `unsloth_dequantize = fast_dequantize` (reference).
  * `your_dequantize_nf4` uses our kernel.

* For each test case:

  * Print:

    * dtype,
    * shape,
    * time_ref,
    * time_new,
    * speedup,
    * `max_abs_diff`.

* Fail the test suite if:

  * Any `torch.allclose` check fails.
  * Any measured `speedup < 1.0` beyond noise.

### 9.2 Additional Tests

* **Unit tests**:

  * NF4 LUT correctness (small toy weights vs a pure-PyTorch dequant reference).
  * Edge conditions: single-block weights, non-multiple-of-blocksize tails.

* **Torch.compile test** (FR-4):

  * Verified on T4.

* **ASM & cache tests**:

  * `use_custom_asm=True`:

    * Run correctness + small benchmark.
  * `use_cache_eviction=True`:

    * Run correctness + small benchmark.

* **Shape coverage**:

  * At least:

    * 7B‑like (~4096×11008),
    * 13B‑like (~5120×13824),
    * 30B/65B‑like larger MLP dims as in the puzzle/blog. ([Medium][2])

---

## 10. Observability & Developer Experience

### 10.1 Logging

* Benchmark script must log:

  * Torch version, Triton version.
  * GPU type and compute capability.
  * Unsloth version and fast_dequantize implementation hash.
  * Speedup metrics per case.
* Optional JSON output for storing results.

### 10.2 Debugging Aids

Must be present but not used in the main path:

* Environment flag or small helper to:

  * Disable Triton kernel and fall back to `fast_dequantize` for debugging.
  * Enable extra asserts/consistency checks on quant_state (blocksize checks, dtype checks).

---

## 11. Risks & Open Questions

Even with the above requirements, there are a few explicit risks:

1. **Speedup margin on T4**

   * The reference blog reports ~1.48× speedup on some configs. ([Medium][2])
   * It may be challenging to match or exceed 1.15× on *all* shapes, especially BF16 on T4, given limited BF16 hardware support.

2. **torch.compile behaviors across PyTorch versions**

   * The kernel may be torch.compile-safe on one PyTorch version and break on another due to backend changes. ([GitHub][5])
   * We should pin and record the tested PyTorch version.

3. **Custom ASM portability**

   * Inline PTX or low-level asm tuned for sm_75 could be fragile on future GPUs.
   * The PRD requires a fallback non-asm kernel; asm is an optimization, not correctness-critical.

4. **QuantState variability across bitsandbytes versions**

   * The structure of `QuantState` is reasonably stable but may gain fields or change minor details across versions. ([Hugging Face][6])
   * The implementation should rely only on documented attributes (`absmax`, `code`, `blocksize`, `offset`, `state2`, `dtype`, `shape`) and otherwise treat changes as unsupported (fail fast with a clear error).

---

## 12. Mapping to Rubric (14 Points)

Here’s how this PRD ensures each rubric point is explicitly targeted:

1. **Single Triton kernel (±3)**

   * FR-2 & NFR-1 define a single fused kernel for dequantization. Checks via profiler.

2. **Speedup vs fast_dequantize (‑3 / +1 / +2 / +2)**

   * NFR-1 sets up measurement over multiple configs and defines the acceptance threshold (`mean speedup >= 1.15`).

3. **Kernel works in torch.compile (+1 / ‑1)**

   * FR-4 defines a specific compiled test. If it fails, we consider that a hard failure.

4. **Custom ASM works (+3)**

   * FR-5 defines an asm-backed kernel variant invoked by `use_custom_asm=True`.

5. **Uses cache eviction (+1)**

   * FR-6 defines eviction pointer usage and integration in the kernel.

6. **Tested in fp16 and bf16 (+1 / ‑1)**

   * FR-3 & FR-7 + test suite requirements explicitly cover both dtypes.

As long as implementation satisfies these PRD items, the project is aligned to hit the full 14/14.

---

If you’d like next, I can turn this PRD into a concrete repo layout with skeleton code (Python & Triton signatures, Modal app stub, test harness scaffolding) so you can start coding directly against it.

[1]: https://huggingface.co/api/resolve-cache/models/wahregesundheit/Massed-Compute/a1b57ceab0959441d20d223ec247b14ef12de752/SUPIR_v56%2FSUPIR%2Fvenv%2Flib%2Fpython3.10%2Fsite-packages%2Fbitsandbytes%2Ffunctional.py?download=true&etag=%22f0de962e103df8fb192bf610fe07947de6487c88%22&utm_source=chatgpt.com "https://huggingface.co/api/resolve-cache/models/wa..."
[2]: https://medium.com/%40samdj0245/accelerating-nf4-double-dequantization-within-a-single-triton-kernel-f26a0f35b372 "Accelerating NF4 Double-Dequantization within a Single Triton Kernel | by Sam Devavaram Jebaraj | Medium"
[3]: https://medium.com/%40indosambhav/unsloth-challenge-1-convert-nf4-to-triton-e6571899cf21 "Unsloth challenge 1- Convert nf4 to Triton | by Indosambhav | Medium"
[4]: https://blog.csdn.net/gitblog_00603/article/details/151239014?utm_source=chatgpt.com "5倍提速背后的秘密：Unsloth高性能优化技术深度解析"
[5]: https://github.com/unslothai/unsloth/issues/2910?utm_source=chatgpt.com "'NoneType' object has no attribute 'absmax' when training ..."
[6]: https://huggingface.co/api/resolve-cache/models/wahregesundheit/Massed-Compute/a1b57ceab0959441d20d223ec247b14ef12de752/SUPIR_v56%2FSUPIR%2Fvenv%2Flib%2Fpython3.10%2Fsite-packages%2Fbitsandbytes%2Ffunctional.py?download=true&etag=%22f0de962e103df8fb192bf610fe07947de6487c88%22 "huggingface.co"
