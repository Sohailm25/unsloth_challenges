# Current State (2025-11-28)
- Branch: chore/split-challenges; ahead of origin.
- Environment: `.venv` Py3.11; Modal images pin torch 2.3.1 / triton 2.3.1 / bitsandbytes 0.43.1 / transformers>=4.41 / peft>=0.11 / trl<0.9; xformers omitted.
- NF4 kernel (`challenges/challenge_a_nf4.py`): fused Triton kernel with constexpr shifts, optional asm/cache, bf16 emulation. Store uses reshape+where to write contiguous 2*BLOCK_SIZE tile. Autotune configs now 256/512/512w8/1024w8. Eviction hints added on absmax/code loads. torch.compile path falls back to fast_dequantize.
- Test status (Modal T4, 2025-11-28): `tests/test_challenge_a_nf4.py` pass (run ap-AEjERuPUdPNm1m3HpMX1eN).
- Benchmark status (Modal T4, 2025-11-28): ref_time=5.24s, new_time=11.79s, speedup≈0.45× (run ap-krRu7nS4KtpVNkdusllx3l); still far from ≥1.15× target.
- Next focus: larger perf gains—try vectorized packed-byte load + asm unpack, possibly higher BLOCK_SIZE (2048) if registers allow, and profile to find hot spot. Consider enabling custom asm path by default if it helps, but avoid fallback “backdoors”.
