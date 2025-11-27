# ABOUTME: Modal entrypoint to run NF4 challenge tests and benchmarks on T4.
# ABOUTME: Builds GPU image and executes pytest/test_dequantize remotely.

import modal


image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime")
    .pip_install(
        "torch==2.3.1",
        "triton==2.3.1",
        "bitsandbytes==0.43.1",
        "transformers>=4.41.0",
        "peft>=0.11.0",
        "unsloth",
        "xformers==0.0.26",
        "pytest",
    )
    .add_local_dir(".", "/workspace")
)


app = modal.App("nf4-challenge")


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
)
def run_tests():
    import os
    import subprocess

    os.chdir("/workspace")
    subprocess.run([
        "pytest",
        "tests/test_challenge_a_nf4.py",
        "-q",
    ], check=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
)
def run_benchmarks():
    import os
    import time
    import torch
    from challenges.challenge_a_nf4 import test_dequantize, your_dequantize_nf4, unsloth_dequantize

    os.chdir("/workspace")
    torch.cuda.synchronize()
    start_ref = time.time()
    ref_time = test_dequantize(unsloth_dequantize)
    torch.cuda.synchronize()
    start_new = time.time()
    new_time = test_dequantize(your_dequantize_nf4)
    torch.cuda.synchronize()
    print({"ref_time": ref_time, "new_time": new_time, "speedup": ref_time / new_time})
