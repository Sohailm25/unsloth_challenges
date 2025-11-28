# Current State (2025-11-28)
- Branch: chore/split-challenges; ahead of origin.
- Environment: `.venv` Py3.11; Modal images pin torch 2.3.1 / triton 2.3.1 / bitsandbytes 0.43.1 / transformers>=4.41 / peft>=0.11 / trl<0.9; xformers omitted.
- NF4 kernel (`challenges/challenge_a_nf4.py`): fused Triton kernel with constexpr shifts, optional cache-evict, bf16 emulation. Vectorized path now loads uint32 (4 packed bytes) per lane (LOAD_VEC=4), decodes 8 nibbles, and uses reshape+where contiguous store; debug path disabled to keep compilation clean. Autotune configs: BLOCK_SIZE 256/512/512@8w/1024@8w (LOAD_VEC fixed at 4).
- Test status (Modal T4, 2025-11-28): `tests/test_challenge_a_nf4.py` pass (run ap-r6aUjeVUevixnuVE2YDpdx).
- Benchmark status (Modal T4, 2025-11-28): ref_time=5.33s, new_time=11.91s, speedup≈0.45× (run ap-z58elecqTMov1cUbxC0XAb). Performance still far below ≥1.15× target; vectorized loads did not help materially.
- Next focus: identify bottleneck (likely store pattern or absmax loads) via profiling; consider reverting to simpler block-sized strided store for correctness and exploring smaller BLOCK_SIZE (128) or higher num_warps, or a hand-inlined asm interleave that avoids reshape overhead. Might need profiler guidance to decide direction.
