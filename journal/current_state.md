# Current State (2025-11-28)
- Branch: chore/split-challenges; working tree contains NF4 kernel and Modal tooling; ahead of origin.
- Environment: `.venv` Python 3.11; Modal images pin torch 2.3.1 / triton 2.3.1 / bitsandbytes 0.43.1 / transformers>=4.41 / peft>=0.11 / trl<0.9; xformers omitted.
- NF4 kernel (`challenges/challenge_a_nf4.py`): fused Triton kernel with constexpr shifts for absmax/absmax2, optional asm nibble unpack + cache-evict load, bf16 emulation on pre-Ampere. Store path now uses reshape+where to build a contiguous 2*BLOCK_SIZE tile and single tl.store (Triton 2.3.1 safe). torch.compile path falls back to fast_dequantize.
- Test status (Modal T4, 2025-11-28): `tests/test_challenge_a_nf4.py` all pass after contiguous store change (run https://modal.com/apps/sohailm25/main/ap-bFKiVbDaT8LzlHXDI0wqbd).
- Benchmark status (Modal T4, 2025-11-28): `run_benchmarks` shows ref_time=5.37s, new_time=11.80s, speedup≈0.46× (run https://modal.com/apps/sohailm25/main/ap-li1xpuN93AXzHEXxQMWG00). Performance improved from 0.36× but still far below ≥1.15× target.
- Next focus: further perf tuning—vectorized packed-byte load + asm nibble unpack, autotune configs including BLOCK_SIZE 1024/num_warps 8, and profiling cache-evict flag. Keep correctness intact while iterating.
