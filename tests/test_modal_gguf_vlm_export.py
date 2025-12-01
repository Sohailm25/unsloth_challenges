"""
ABOUTME: Tests torchao stub toggle used by modal GGUF vision export helper.
ABOUTME: Verifies stub is injected only when env flag is enabled.
"""

import importlib
import os
import sys


def test_torchao_stub_enabled(monkeypatch):
    # Ensure clean slate.
    for mod in [
        "torchao",
        "torchao.quantization",
        "torchao.dtypes",
        "torch._inductor.custom_graph_pass",
    ]:
        sys.modules.pop(mod, None)

    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCHAO", "1")

    helper = importlib.import_module("modal_gguf_vlm_export")
    helper._maybe_disable_torchao()

    assert "torchao" in sys.modules
    assert "torchao.quantization" in sys.modules
    assert "torchao.dtypes" in sys.modules
    assert "torch._inductor.custom_graph_pass" in sys.modules


def test_torchao_stub_disabled(monkeypatch):
    for mod in [
        "torchao",
        "torchao.quantization",
        "torchao.dtypes",
        "torch._inductor.custom_graph_pass",
    ]:
        sys.modules.pop(mod, None)

    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCHAO", "0")

    helper = importlib.import_module("modal_gguf_vlm_export")
    helper._maybe_disable_torchao()

    assert "torchao" not in sys.modules
    assert "torchao.quantization" not in sys.modules
    assert "torchao.dtypes" not in sys.modules
    assert "torch._inductor.custom_graph_pass" not in sys.modules
