# Current State (2025-11-27)
- Branch: chore/split-challenges; uncommitted changes include NF4 kernel implementation, tests, modal entrypoint, research log, and pre-commit config.
- challenge_a_nf4.py now contains a fused Triton NF4 dequant kernel with shift-based indexing, LUT caching, optional asm and cache-eviction flags, and wrapper wiring.
- tests/test_challenge_a_nf4.py exercises correctness (fp16/bf16), torch.compile path, and optional flags; CUDA-only, to be run via Modal.
- modal_challenge_a.py provides T4 execution for pytest and benchmark comparison against unsloth fast_dequantize.
- research/oracle_list-concrete-implementa-tips-to.md captures Triton tuning guidance; PRD remains in reference/full_prd.md.
- Pre-commit hooks configured in .pre-commit-config.yaml; hooks not yet run locally.
- No GPU-backed tests executed locally; Modal run pending to validate correctness and performance.
