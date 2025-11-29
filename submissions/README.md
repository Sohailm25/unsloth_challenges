# Submission Bundle

This folder collects the files a maintainer would review for scoring. Files are symlinks to the originals to avoid divergence.

## Challenge A (NF4 Triton)
- `challenge_a_nf4.py` – final kernel implementation. Test: `pytest tests/test_challenge_a_nf4.py -q`. Benchmark: see `logs/2025-11-29-part-a.log` for Part A integration timing vs B baseline, and re-run `modal_challenge_a.py::run_benchmarks` for speedup >=1.15x on T4.

## Challenge C (torch.compile QLoRA)
- `challenge_c_solution.py` – compiled MLP/attention/RMSNorm/loss with optional Part A custom_op. Run: `python -m challenges.challenge_c_solution` (set `TORCH_LOGS=graph_breaks,recompiles`). Dynamic shapes via flex_attention on sm80+.

## Challenge E (Memory-efficient backprop)
- `challenge_e_solution.py` – chunked forward/backward autograd Function. Run: `python -m challenges.challenge_e_solution`; modal VRAM check: `modal run modal_challenge_e.py::test_memory_reduction`.

## Challenge B (FSDP2 + QLoRA) – In Progress
- `challenge_b_train.py` – baseline FSDP2 + QLoRA (+ optional torch.compile). 
- `challenge_b_train_with_part_a.py` – experimental Part A kernel integration (currently slower/worse loss under FSDP2).
- Kaggle assets: `kaggle_challenge_b_fsdp2_qlora.ipynb`, `kaggle_challenge_b_fsdp2_qlora.py`.
- Benchmarks: `logs/2025-11-29-bnb-baseline.log` (BnB) and `logs/2025-11-29-part-a.log` (Part A kernel); Part A presently regressions in loss and speed.

## How to re-run key checks
- NF4 kernel: `pytest tests/test_challenge_a_nf4.py -q`.
- Challenge C graph-break audit: `TORCH_LOGS=graph_breaks,recompiles TORCHDYNAMO_VERBOSE=1 python -m challenges.challenge_c_solution`.
- Challenge E self-tests: `python -m challenges.challenge_e_solution`.
- FSDP2 baseline (2x T4 via Modal): `modal run modal_challenge_b.py::run_fsdp_training --use_fsdp2 --use_torch_compile --max-steps 60`.
