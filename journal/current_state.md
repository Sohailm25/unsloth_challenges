# Current State (2025-11-28)
- Branch: chore/split-challenges; ahead of origin.
- Environment: `.venv` Py3.11; Modal images pin torch 2.3.1 / triton 2.3.1 / bitsandbytes 0.43.1 / transformers>=4.41 / peft>=0.11 / trl<0.9; xformers omitted.
- NF4 kernel (`challenges/challenge_a_nf4.py`): fused Triton kernel with constexpr shifts, optional cache-evict, bf16 emulation. Vectorized LOAD_VEC=4 path (uint32 load -> 8 nibbles) and contiguous reshape+where store. Debug path removed in kernel. Autotune configs now 128/256/512/512@8w (LOAD_VEC fixed at 4).
- Test status (Modal T4, 2025-11-28): `tests/test_challenge_a_nf4.py` pass (ap-dtK4HK7L43zdHlW43mYYzx).
- Benchmark status (Modal T4, 2025-11-28): ref_time≈5.30s, new_time≈11.89s, speedup≈0.45× (ap-KeYqW92CRc5d5uNXVE2AvU). Performance remains far below ≥1.15× target; vectorization and block-size tuning did not help.
- Next focus: profile to locate bottleneck (stores vs absmax loads); consider alternative store scheme or reducing reshape overhead; possibly try LOAD_VEC=2 variant or shared-memory caching of absmax/absmax2 if profiling shows bandwidth bound.
