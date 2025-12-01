## Current state as of 2025-12-01

- GGUF vision export validated for both families on Modal:
  - Qwen/Qwen-VL (run ap-0zteD8dLwAKt9ZgmXnY3XV, torch2.9 image) produced `Qwen-VL.BF16.gguf`, `Qwen-VL.BF16-mmproj.gguf`, and `Qwen-VL.Q8_0.gguf`.
  - LLaVA 1.5 7B (run ap-3U78P5zbWbLg0HDk2eKxUD, torch2.9 image, detached) produced `llava-1.5-7b-hf.BF16.gguf`, `llava-1.5-7b-hf.BF16-mmproj.gguf`, and `llava-1.5-7b-hf.Q8_0.gguf`; example usage: `llama-mtmd-cli -m llava-1.5-7b-hf.Q8_0.gguf --mmproj llava-1.5-7b-hf.BF16-mmproj.gguf`.
- `unsloth/gguf_vlm_helpers.py` patches (committed earlier) handle VLM detection, skip vision tensors for text export, tokenizer trust_remote_code, Llava clip acceptance, and Qwen2-VL config normalization (auto preprocessor_config).
- `modal_gguf_vlm_export.py` now sets torchao-disabling env vars and injects a torchao stub so transformer quantizers cannot crash torch 2.5/2.9 images; default runs use torch 2.9 image for VLMs.
- Tests: `tests/saving/vision_models/test_vlm_gguf_helpers.py` remain green; no new local runs today.
- Next steps: optionally pull modal artifacts locally/publish, document usage in repo, and ensure pre-commit hooks generated before shipping.
- Part A + FSDP2 path: added shape-bounded debug prints, sharded quant_state slicing passthrough, row-shard unit test, and `__getattr__` passthroughs for `_ShardedQuantState/State2`; row-shard path now uses BnB dequant with local-row slicing and gathers outputs.
- Modal TinyLlama 1-step smokes still failing previously (shape mismatches and attr errors); latest run ap-ImgGDRUdPwVtezvPbTIQZA is still executing with row-sliced gather (shape logs show weight (1024,2048) per rank). Need to re-poll and verify completion before proceeding to 8B runs.
