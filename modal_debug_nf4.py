# ABOUTME: Modal entrypoint to run a minimal NF4 debug comparison on T4.
# ABOUTME: Prints per-block dequant values to pinpoint mismatches.

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.3.0",
        "triton==2.3.0",
        "bitsandbytes==0.43.1",
        "transformers>=4.41.0",
        "peft>=0.11.0",
        "unsloth",
        "xformers==0.0.26.post1",
        "trl<0.9.0",
    )
    .add_local_dir(".", "/workspace")
)

app = modal.App("nf4-debug")


@app.function(image=image, gpu="T4", timeout=600)
def debug_block():
    import os
    import torch
    from transformers import set_seed
    from challenges.challenge_a_nf4 import MLP, _debug_dequant_single_block

    os.chdir("/workspace")
    set_seed(3407)
    torch.set_default_dtype(torch.float32)
    mlp = MLP(hd=128, m=256, dtype=torch.float16)
    info = _debug_dequant_single_block(mlp.up_proj)
    torch.set_printoptions(sci_mode=False, linewidth=120, precision=6)
    print("packed:", info["packed"][:8])
    print("absmax_codes:", info["absmax_codes"][:8])
    print("absmax2:", info["absmax2"][:8])
    print("offset:", info["offset"])
    print("n_weights:", info["n_weights"], "n_bytes:", info["n_bytes"], "blocksize:", info["blocksize"], "blocksize2:", info["blocksize2"])
    print("host_out first 8:", info["host_out"][:8])
    print("kernel_out first 8:", info["kernel_out"][:8])
    diff = (info["host_out"][:128] - info["kernel_out"][:128]).abs()
    print("max_diff_first_block:", diff.max())
