"""
ABOUTME: Modal helper to validate GGUF export for vision models (Llava/Qwen VL).
ABOUTME: Builds an image with local unsloth code and runs save_pretrained_gguf on a small VLM.
"""

from modal import App, Image

stub = App("gguf-vlm-export-test")

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "libcurl4-openssl-dev", "curl")
    .pip_install(
        [
            "torch==2.5.1",
            "torchvision==0.20.1",
            "torchaudio==2.5.1",
            "torchao==0.13.0",
            "sentencepiece",
            "accelerate",
            "bitsandbytes",
            "unsloth_zoo==2025.11.5",
        ],
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(
        "./unsloth",
        "/workspace/unsloth",
        ignore=[".venv", "__pycache__", ".git"],
        copy=True,
    )
)


@stub.function(image=image, gpu="A10G", timeout=60 * 60)
def run(model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
    import os
    import subprocess
    import sys

    # Install local unsloth in editable mode so our branch code is used.
    subprocess.run(["pip", "install", "-e", "/workspace/unsloth"], check=True)
    sys.path.insert(0, "/workspace/unsloth")

    # Shim uv -> pip to avoid uv virtualenv requirement inside install_llama_cpp.
    shim_dir = "/workspace/shim"
    os.makedirs(shim_dir, exist_ok=True)
    uv_path = os.path.join(shim_dir, "uv")
    with open(uv_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write('if [ "$1" = "pip" ]; then shift; exec pip "$@"; else exec /usr/bin/env uv "$@"; fi\n')
    os.chmod(uv_path, 0o755)
    os.environ["PATH"] = f"{shim_dir}:" + os.environ["PATH"]
    os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

    import torch
    # Torch 2.5 lacks torch.int1 required by torchao; stub to int8 for import-time checks.
    for n in range(1, 9):
        if not hasattr(torch, f"int{n}"):
            setattr(torch, f"int{n}", torch.int8)
        if not hasattr(torch, f"uint{n}"):
            setattr(torch, f"uint{n}", torch.uint8)

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True,
    )

    base_dir = "/workspace/out_model"
    os.makedirs(base_dir, exist_ok=True)
    model.save_pretrained(base_dir, tokenizer=tokenizer)
    tokenizer.save_pretrained(base_dir)

    model.save_pretrained_gguf(
        base_dir,
        tokenizer=tokenizer,
        quantization_method=["fast_quantized"],
        push_to_hub=False,
    )

    print("Saved files:", os.listdir(base_dir))
    for fname in os.listdir(base_dir):
        if fname.endswith(".gguf"):
            fpath = os.path.join(base_dir, fname)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"{fname}: {size_mb:.1f} MB")


if __name__ == "__main__":
    stub.cli()
