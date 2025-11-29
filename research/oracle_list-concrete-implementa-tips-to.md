## description of what was run, for what purpose
- Oracle run (browser engine) to gather actionable implementation tips for NF4 fused Triton dequant kernel tuned for Tesla T4, including BLOCK_SIZE/num_warps, byte-unpack patterns (with tl.asm), cache-eviction, and bf16 pitfalls.

## prompt
List concrete, implementable tips to build the NF4 fused Triton dequant kernel for Tesla T4: best BLOCK_SIZE/num_warps choices, byte-unpack patterns (including tl.asm examples), cache-eviction pattern, and bf16 handling pitfalls. Keep it concise and actionable.

## files provided to oracle
- reference/full_prd.md
- reference/engineer_plan.md
- challenges/challenge_a_nf4.py

## key takeaways
- Autotune set: BLOCK_SIZE {256/4w, 512/4w, 512/8w, 1024/8w}; default 512/4w; keep BLOCK_SIZE multiple of 128; num_stages 2–3.
- Use log2 ratios to avoid div/mod: pass offset1=log2(blocksize), offset2=log2(blocksize2); compute absmax_idx via shifts.
- Nibble extraction: pure Triton bit ops on uint32; optional tl.asm snippet shown for hi/lo; optional uint32 vector load + asm to unpack 4 bytes.
- Coalesced store: build 2*BLOCK_SIZE tile with tl.where(is_hi, weight_h[idx], weight_l[idx]) and single contiguous tl.store.
- Cache eviction rubric: optional evict_ptr load with eviction_policy="evict_last" gated by USE_CACHE_EVICT flag.
- BF16 on T4: compute in fp32; specialize kernel per OUT_DTYPE (fp16/bf16) and cast at store; keep absmax/absmax2/code2 in fp32; avoid extra fp16->bf16 conversion kernel.
- NF4 LUT: keep small fp32 tensor, tl.load; use tl.fma(code_val, scale, offset) for absmax reconstruction.
