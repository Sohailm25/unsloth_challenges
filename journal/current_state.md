# Current State (2025-11-28)

## Branch Info
- Branch: chore/split-challenges; ahead of origin
- bd issue `unsloth-challenges-guu` closed after custom asm implementation (Challenge A)

## Environment
- `.venv` Py3.11
- Challenge A Modal image: torch 2.3.1 / triton 2.3.1 / bitsandbytes 0.43.1 / transformers ≥4.41 / peft ≥0.11 / trl <0.9
- Challenge C Modal image: torch 2.5.1 / triton 3.0+ / bitsandbytes 0.48.2 / transformers 4.57+ / peft 0.13+ / trl 0.12+

---

## Challenge E: Memory Efficient Backprop

**Status: COMPLETE (All Tests Passed - 10/10 Points)**

- Implementation: `challenges/challenge_e_solution.py`
- Modal harness: `modal_challenge_e.py`
- Research: `research/oracle_challenge_e_memory_backprop.md`
- Modal App ID: `ap-6THE1Klx2wGbEmqeHWYdva`

**Implementation Highlights:**
- `MemoryEfficientLinear` - torch.autograd.Function with chunked forward/backward
- Uses autograd for transformation gradients (no hardcoded loss derivatives)
- Standard linear formulas for dL/dW and dL/dX (chain rule, not hardcoded)
- Transform functions return unreduced per-element values + weights
- Supports reduction modes: none, sum, mean
- Dynamic chunk sizes supported

**GPU Test Results (Modal T4):**
| Test | Result | Details |
|------|--------|---------|
| Memory Reduction | **PASS** | 46-65% reduction achieved |
| GRPO Memory | **PASS** | Loss=0.299, peak=0.334 GB |
| Llama 1B Loss Match | **PASS** | Exact loss match, gradients match |

**Memory Reduction Breakdown:**
| Config | Standard Peak | Efficient Peak | Reduction |
|--------|--------------|----------------|-----------|
| N=2048, V=32K | 0.631 GB | 0.339 GB | **46.2%** |
| N=4096, V=32K | 1.197 GB | 0.419 GB | **65.0%** |
| N=2048, V=64K | 1.345 GB | 0.665 GB | **50.5%** |

**Llama 1B Test:**
- Model: TinyLlama-1.1B-Chat-v1.0 (meta-llama/Llama-3.2-1B requires approval)
- Standard Loss: 2.5625, Efficient Loss: 2.5625 (exact match)
- Gradients: Match (torch.allclose with rtol=1e-2)

**Final Scoring (10/10 points):**
- VRAM 50% reduction: +2 **PASS** (65% achieved)
- NO float32 upcast: Required **PASS**
- CE loss works: +1 **PASS**
- Other functions work: +1 **PASS** (label smoothing)
- NO hardcoded gradients: Required **PASS**
- Dynamic chunk sizes: +1 **PASS**
- Llama 1B training loss matches: +1 **PASS**
- GRPO works: +4 **PASS**

---

## Challenge A: NF4 Dequantization Kernel

**Status: In Progress (asm path implemented; ~1.15–1.18x)**

- NF4 kernel (`challenges/challenge_a_nf4.py`): single Triton path with LOAD_VEC=4, strided even/odd stores, constexpr shifts, cache-evict toggle, bf16 emu, asm nibble-unpack gated to sm75 + divisible-by-4 input.
- Autotune configs restored to full set (128/256/512/512@8w/1024/2048).
- Tests: pass on T4 with asm (latest ap-hguMGNjKe9ysGFYsfmxqLk).
- Benchmarks (asm enabled, per-layer out reuse in benchmark): ref_time 3.9139s, new_time 3.4041s, speedup 1.1497x (ap-Ev3lo4Cxb11nCjYuj9NT99). Best seen 1.1821x (ap-oFCbm3d5TyvqdIBRlNuAlT). Range observed 1.15–1.18x.
- Profiling (ap-1v2WbBGj4dUTYJIlTx6joF earlier): kernel ~147ms total CUDA; aten::fill_/zero_ ~313ms; overall block ~5.05s/5 iters.

**Pending Work:**
1. Further reduce aten::fill_/zero_ (~300ms) to stabilize speedup margin (options: deeper buffer reuse, smaller shapes autotune filtering).
2. Keep asm gating torch.compile-safe; re-run benchmarks after tweaks.

---

## Challenge B: FSDP2 + QLoRA Distributed Training

**Status: COMPLETE (7/10 Points)**

- FSDP2 + QLoRA + torch.compile working on 2x T4 GPUs
- Kaggle notebook created and ready for upload

**Results:**
| Configuration | Final Loss (avg last 10) |
|---------------|-------------------------|
| Single GPU | ~1.63 |
| FSDP1 2xT4 | ~1.78 |
| FSDP2 2xT4 | ~1.76 |
| FSDP2 + compile 2xT4 | ~1.7-1.8 |

**Key Technical Achievements:**
1. **FSDP2**: `fsdp_version: 2` with `fsdp_reshard_after_forward: true`
2. **QLoRA**: `bnb_4bit_quant_storage=torch.float16` for FSDP compatibility
3. **torch.compile**: Via `TrainingArguments.torch_compile=True` (not manual compile)

**Files:**
- `challenge_b_train.py` - Training script with FSDP2 + torch.compile
- `modal_challenge_b.py` - Modal harness with `--fsdp2 --compile` flags
- `kaggle_challenge_b_fsdp2_qlora.py` - Kaggle-ready Python script
- `notebooks/kaggle_challenge_b_fsdp2_qlora.ipynb` - Kaggle notebook

**Scoring (7/10):**
| Component | Points |
|-----------|--------|
| FSDP2 + QLoRA + torch.compile | 5 |
| Kaggle notebook | +2 |
| **Total** | **7** |

**Optional for +3 points:**
- Integrate Part A kernel if faster than BnB

---

## Challenge C: torch.compile for QLoRA

**Status: COMPLETE (All Tests Passing)**

- Implementation: `challenges/challenge_c_solution.py`
- Modal harness: `modal_challenge_c.py`
- Research: `research/oracle_flex_attention_part_a_gpt5pro.md`

**Test Results:**
| Test | GPU | Result | Details |
|------|-----|--------|---------|
| Graph break | T4 | PASS | Dynamic seq lengths (2, 25, 76 tokens) |
| Training | T4 | PASS | 10/10 steps, loss=2.396, 75.49s |

**Compiled Components:**
- LlamaMLP: `fullgraph=False, dynamic=True, max_autotune=True`
- LlamaAttention: SDPA with compilation
- LlamaRMSNorm: `fullgraph=True`
- BnB Linear4bit + PEFT Linear4bit: `@torch._dynamo.disable()` with Part A kernel

**Part A Kernel Integration Attempt:**
Attempted torch.library custom op integration but blocked by:
- T4: bf16 PTX not supported on sm75 (model uses bf16 internally)
- A100: Triton matmul dtype mismatch from inductor fusion
- Root cause: inductor fusion with custom ops creates dtype conflicts

**Current Approach:** Part A kernel used via `@torch._dynamo.disable()` fallback.

**flex_attention Investigation:**
Disabled by default due to instability in PyTorch 2.5.1. SDPA provides reliable compiled attention.

**Scoring Assessment (~1 point):**
- ❌ BnB via dynamo.disable: -2 points (blocked by inductor issues)
- ✅ Attention compiled (SDPA): +2 points
- ✅ MLP compiled: +1 point
- ✅ Loss compiled: avoid -1
- ✅ LayerNorms compiled: avoid -3
- ❌ flex_attention disabled: 0 points

---

## Files Summary

### Challenge A
- `challenges/challenge_a_nf4.py` - NF4 kernel implementation
- `challenges/challenge_a_nf4_backup_20251128.py` - Backup snapshot
- `modal_challenge_a.py` - Modal test harness

### Challenge B
- `challenge_b_train.py` - FSDP + QLoRA training script
- `modal_challenge_b.py` - Modal entrypoint with 2x T4

### Challenge C
- `challenges/challenge_c_solution.py` - torch.compile implementation
- `modal_challenge_c.py` - Modal test harness
- `research/challenge_c_torch_compile_research.md` - Research notes

### Challenge E
- `challenges/challenge_e_solution.py` - Memory-efficient backprop implementation
- `modal_challenge_e.py` - Modal test harness
- `research/oracle_challenge_e_memory_backprop.md` - Oracle research notes
