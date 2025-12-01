## Current state as of 2025-11-30

- GGUF vision export: Qwen/Qwen-VL now converts successfully on Modal (run ap-0zteD8dLwAKt9ZgmXnY3XV). Outputs: `Qwen-VL.BF16.gguf`, `Qwen-VL.BF16-mmproj.gguf`, and quantized `Qwen-VL.Q8_0.gguf`. mmproj conversion completes after injected guard sets `mapped = None`.
- Helper patches in `unsloth/gguf_vlm_helpers.py`: VLM detection/aliases, skip vision tensors for text export, tokenizer trust_remote_code, Llava clip acceptance, Qwen2-VL config normalization with default `preprocessor_config.json`.
- Unit coverage: `tests/saving/vision_models/test_vlm_gguf_helpers.py` all passing.
- Remaining work: consider running Llava modal export to verify both VLM families; optionally fetch modal artifacts locally for validation; then commit and push.
