# Oracle: disable torchao imports in transformers 4.57 (2025-12-01)

**Prompt:** How do I prevent transformers 4.57 quantizer modules from importing torchao when torch_ao dependencies are missing (`ModuleNotFoundError: torch._inductor.custom_graph_pass`)? Prefer environment flags or stubbing approaches that work on torch 2.5/2.9 images.

**Key takeaways (gpt-5-pro):**
- No official Transformers env flag toggles torchao imports; torchao is optional and only needed with `TorchAoConfig`.
- Import failure arises because torchao imports PyTorch inductor internals; some torch builds lack `torch._inductor.custom_graph_pass`.
- Recommended mitigation: pre-load a stub `torchao` module so transformers never imports the real package. Implement via `sitecustomize.py` or per-process shim; gate with `TRANSFORMERS_DISABLE_TORCHAO=1`.
- Stub should include `torchao`, `torchao.quantization`, `torchao.dtypes` modules and optionally stub `torch._inductor.custom_graph_pass` to satisfy imports. If any torchao API is invoked, raise `ImportError` for clarity.
- Absolute fallback: uninstall torchao from the image.

**Suggested action for modal runs:**
- Drop a `sitecustomize.py` stub (or add the per-process shim) before importing transformers/unsloth when running torch 2.5 images that ship torchao without inductor. Set `TRANSFORMERS_DISABLE_TORCHAO=1` in the modal function env.

**Session:** how-do-i-prevent-transforme-2 (gpt-5-pro, 10m48s)
