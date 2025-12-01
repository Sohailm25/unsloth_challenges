## Unified Attention Refactor Summary

Scope: merge SDPA, xFormers, FlashAttention-2, and FlexAttention behind a single interface for Unsloth models, with capability-aware selection and explicit overrides.

### What was implemented
- Added `unsloth/attn/` with a unified API (`unified_attention`, `pick_backend`, `detect_capabilities`) and backend adapters for SDPA, xformers, flash-attn2, flex.
- Patched model attention calls (LLaMA, Mistral, Qwen3, Gemma2, Granite, Cohere, Falcon H1, vision) to route through `unified_attention`; LLaMA passes FlexSpec when masks require it.
- Capability gating: flash-attn disabled on compute capability < 8; xformers/flex detected optionally; env override `UNSLOTH_ATTENTION` respected for debugging.
- Tests: `tests/test_attention_backend_selection.py` (backend choice, SM75 fallback, flash stub), `tests/test_unified_attention_smoke.py` (CPU parity, masks, grads).

### Validation
- A100 Modal smokes: backend selected `flash_attn_2`, gradients OK (`runs/benchmarks/2025-11-30-ampere-flash-attn-model-smoke.log`).
- Perf sweeps on A100 (torch 2.5.1+cu121, flash-attn 2.8.3): logs `runs/benchmarks/2025-11-30-ampere-flash-attn-perf*.log`, plus broad grid `...-perf-grid.log`. Flash wins for long sequences (≈≥2K tokens); SDPA faster on short seq; some head configs still favor SDPA.
- T4/SM75: FA2 unsupported; forced fallback verified by unit tests and documented. FA2 path reachability proven via stub test.

### How to control backend
- Default auto prefers flash on Ampere when available; SDPA otherwise. Flex selected when explicitly requested via FlexSpec.
- Override with `UNSLOTH_ATTENTION` = `flash` | `sdpa` | `xformers` | `flex` (subject to capability gates).

### Files of interest
- Code: `unsloth/attn/api.py` and backend adapters `_sdpa.py`, `_xformers.py`, `_flash_attn.py`, `_flex.py`.
- Tests: `tests/test_attention_backend_selection.py`, `tests/test_unified_attention_smoke.py`.
- Modal scripts: `modal_attention_flash_ampere.py` (smoke, perf, grid, model smoke).
- Docs: README note in `unsloth/README.md` under “FlashAttention support”.
