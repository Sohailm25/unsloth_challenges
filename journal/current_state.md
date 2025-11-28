# Current State (2025-11-28)
- Branch: chore/split-challenges; ahead of origin.
- Environment: `.venv` Py3.11; Modal images pin torch 2.3.1 / triton 2.3.1 / bitsandbytes 0.43.1 / transformers>=4.41 / peft>=0.11 / trl<0.9; xformers omitted.
- NF4 kernel (`challenges/challenge_a_nf4.py`): vectorized LOAD_VEC=4, strided even/odd stores, constexpr shifts, optional cache-evict, bf16 emu. Autotune configs 128/256/512/512@8w/1024/2048. Device caching added for absmax/code tensors; output/evict cached.
- Tests: pass on T4 (latest ap-hWOv7hJAsx9a5AZRfuIBie).
- Benchmarks: still slow; latest ref_time≈5.32s, new_time≈12.26s, speedup≈0.43x (ap-jj2cBM3yCy1G0ClEegExHZ). Prior runs hover 0.43–0.48x.
- Profiling: ours kernel ~0.33–0.41 ms/launch; PyTorch-side aten::fill_/zero_ ~0.43s per run; overall ours_kernel block ~3.8–6.1s for 5 iters. Host overhead dominates; kernel not bottleneck.
- Pending work: redesign wrapper to avoid any implicit fills/reshapes and possibly use a functional `torch.empty` output-only path (no caching) or a custom op with explicit scratch to keep torch.compile-safe and eliminate fill/zero ops. Need to re-profile after host refactor.
