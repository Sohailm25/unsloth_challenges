# GGUF Vision Export (LLaVA + Qwen-VL)

## What we changed
- Patched `unsloth/gguf_vlm_helpers.py` to detect vision LMs, skip vision tensors when writing text GGUF, trust remote tokenizers, accept LLaVA clip vision configs, and normalize Qwen2-VL configs (auto-writes `preprocessor_config.json`).
- Added guards in the llama.cpp converter patch so mmproj export no longer crashes when vision tensors are absent during text-only conversion.
- Added modal helper `modal_gguf_vlm_export.py` to run end-to-end GGUF exports on GPU, plus a torchao-stub/env toggle to avoid `torch._inductor.custom_graph_pass` import failures on torch 2.5/2.9 images.
- Unit coverage: `tests/saving/vision_models/test_vlm_gguf_helpers.py` exercises the helper logic; pre-commit runs `tests/test_part_a_fsdp_paths.py`.

## How to run exports on Modal
- Qwen-VL (torch 2.9 image):
  `modal run modal_gguf_vlm_export.py::run_torch29 --model-id Qwen/Qwen-VL`
- LLaVA 1.5 7B (torch 2.9 image):
  `modal run modal_gguf_vlm_export.py::run_torch29 --model-id llava-hf/llava-1.5-7b-hf`

Outputs per run: BF16 text GGUF, BF16 mmproj GGUF, and Q8_0 quantized GGUF. Use with llama.cpp mtmd:
`llama-mtmd-cli -m <model>.Q8_0.gguf --mmproj <model>.BF16-mmproj.gguf`

## Verified runs
- Qwen-VL: Modal run `ap-0zteD8dLwAKt9ZgmXnY3XV` produced `Qwen-VL.BF16.gguf`, `Qwen-VL.BF16-mmproj.gguf`, `Qwen-VL.Q8_0.gguf`.
- LLaVA 1.5 7B: Modal run `ap-3U78P5zbWbLg0HDk2eKxUD` produced `llava-1.5-7b-hf.BF16.gguf`, `llava-1.5-7b-hf.BF16-mmproj.gguf`, `llava-1.5-7b-hf.Q8_0.gguf`.

## Notes
- Keep `TRANSFORMERS_DISABLE_TORCHAO=1` (set in helper) on torch 2.5/2.9 images to prevent torchao import issues; harmless on torch builds with proper torchao.
- Artifacts currently live on Modal volumes; pull or publish as needed.
