# ABOUTME: Modal entrypoint to run a minimal NF4 debug comparison on T4.
# ABOUTME: Prints per-block dequant values to pinpoint mismatches.

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.3.1",
        "triton==2.3.1",
        "bitsandbytes==0.43.1",
        "transformers>=4.41.0",
        "peft>=0.11.0",
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
    print("ref_out first 8:", info["ref_out"][:8])
    diff = (info["host_out"][:128] - info["kernel_out"][:128]).abs()
    print("max_diff_first_block:", diff.max())
    diff_ref = (info["ref_out"][:128] - info["kernel_out"][:128]).abs()
    print("max_diff_kernel_vs_ref_first_block:", diff_ref.max())
    from challenges.challenge_a_nf4 import unsloth_dequantize, your_dequantize_nf4
    full_kernel = your_dequantize_nf4(mlp.up_proj, use_custom_asm=False, use_cache_eviction=False, use_optimized=True)
    full_ref = unsloth_dequantize(mlp.up_proj)
    full_diff = (full_kernel - full_ref).abs()
    max_val, max_idx = full_diff.max(dim=0)
    max_flat = full_diff.view(-1).argmax()
    print("full_max_diff:", full_diff.max().item(), "mismatches >", (full_diff > 1e-5).sum().item())
    print("max_diff_index_flat:", max_flat.item(), "kernel:", full_kernel.view(-1)[max_flat].item(), "ref:", full_ref.view(-1)[max_flat].item())


@app.function(image=image, gpu="T4", timeout=900)
def debug_case():
    import os
    import torch
    from transformers import set_seed
    from challenges.challenge_a_nf4 import MLP, unsloth_dequantize, your_dequantize_nf4

    os.chdir("/workspace")
    set_seed(3407)
    torch.set_default_dtype(torch.float32)
    hd, m, dt = 2048, 8192, torch.float16
    mlp = MLP(hd=hd, m=m, dtype=dt)
    target = unsloth_dequantize(mlp.up_proj)
    ours = your_dequantize_nf4(mlp.up_proj, use_custom_asm=False, use_cache_eviction=False, use_optimized=True)
    diff = (target - ours).abs()
    max_diff = diff.max().item()
    mismatches = (diff > 1e-5).sum().item()
    idx = diff.view(-1).argmax().item()
    qs = mlp.up_proj.weight.quant_state
    blocksize = int(getattr(qs, "blocksize", 64))
    block_start = (idx // blocksize) * blocksize
    block_end = block_start + blocksize
    byte_start = block_start // 2
    block_bytes = blocksize // 2
    packed_slice = mlp.up_proj.weight.data.flatten()[byte_start:byte_start + block_bytes]
    absmax_idx = block_start // blocksize
    absmax_code = qs.absmax[absmax_idx].to(torch.int64)
    code_val = qs.state2.code[absmax_code].to(torch.float32)
    scale = qs.state2.absmax[absmax_idx // qs.state2.blocksize].to(torch.float32)
    absmax = code_val * scale + qs.offset
    manual_absmax = absmax.item()
    hi = (packed_slice >> 4).to(torch.int64)
    lo = (packed_slice & 0x0F).to(torch.int64)
    w_hi = qs.code[hi].to(torch.float32) * absmax
    w_lo = qs.code[lo].to(torch.float32) * absmax
    manual_block = torch.empty(blocksize, device=packed_slice.device, dtype=qs.dtype)
    manual_block[0::2] = w_hi
    manual_block[1::2] = w_lo
    print("packed_block_head:", packed_slice[:8].tolist())
    print({
        "hd": hd,
        "m": m,
        "dtype": str(dt),
        "n_weights": qs.shape.numel() if hasattr(qs, "shape") else None,
        "n_bytes": mlp.up_proj.weight.data.numel(),
        "code2_len": qs.state2.code.numel(),
        "absmax_len": qs.absmax.numel(),
        "absmax2_len": qs.state2.absmax.numel(),
        "max_diff": max_diff,
        "mismatches_gt_1e-5": mismatches,
        "max_idx": idx,
        "block_start": block_start,
        "target_val": target.view(-1)[idx].item(),
        "ours_val": ours.view(-1)[idx].item(),
        "target_block_head": target.view(-1)[block_start:block_start + 8].tolist(),
        "ours_block_head": ours.view(-1)[block_start:block_start + 8].tolist(),
        "manual_block_head": manual_block[:8].tolist(),
    })

    # capture kernel internals for this block
    debug_pid = (block_start // 2) // 256
    debug_out, debug_buf = your_dequantize_nf4(
        mlp.up_proj,
        use_custom_asm=False,
        use_cache_eviction=False,
        use_optimized=True,
        debug_block=debug_pid,
    )
    from challenges.challenge_a_nf4 import _compute_shift_offsets
    print("shifts:", _compute_shift_offsets(mlp.up_proj.weight.data, qs))
    print("debug_block_hi[:8]:", debug_buf[0, :8].tolist())
    print("debug_block_lo[:8]:", debug_buf[1, :8].tolist())
    print("debug_block_absmax[:8]:", debug_buf[2, :8].tolist())
    print("debug_block_hi_indices[:8]:", debug_buf[3, :8].tolist())
    print("debug_block_absmax_idx[:8]:", debug_buf[4, :8].tolist())
    print("lut_head:", qs.code[:8].tolist())
    print("manual_absmax:", manual_absmax)
    print("state2_blocksize:", qs.state2.blocksize)


@app.function(image=image, gpu="T4", timeout=900)
def debug_case_bf16():
    import os
    import torch
    from transformers import set_seed
    from challenges.challenge_a_nf4 import MLP, unsloth_dequantize, your_dequantize_nf4

    os.chdir("/workspace")
    set_seed(3409)
    torch.set_default_dtype(torch.float32)
    hd, m, dt = 1024, 4096, torch.bfloat16
    mlp = MLP(hd=hd, m=m, dtype=dt)
    target = unsloth_dequantize(mlp.up_proj)
    ours = your_dequantize_nf4(mlp.up_proj, use_custom_asm=False, use_cache_eviction=False, use_optimized=True)
    diff = (target - ours).abs()
    max_diff = diff.max().item()
    mismatches = (diff > 1e-5).sum().item()
    idx = diff.view(-1).argmax().item()
    print({"hd": hd, "m": m, "dtype": str(dt), "max_diff": max_diff, "mismatches_gt_1e-5": mismatches, "max_idx": idx, "target_val": target.view(-1)[idx].item(), "ours_val": ours.view(-1)[idx].item()})


@app.function(image=image, gpu="T4", timeout=900)
def debug_all_layers():
    import os
    import torch
    from transformers import set_seed
    from challenges.challenge_a_nf4 import MLP, unsloth_dequantize, your_dequantize_nf4

    os.chdir("/workspace")
    set_seed(3407)
    torch.set_default_dtype(torch.float32)
    hd, m, dt = 2048, 8192, torch.float16
    mlp = MLP(hd=hd, m=m, dtype=dt)
    results = {}
    for name, layer in [("up", mlp.up_proj), ("gate", mlp.gate_proj), ("down", mlp.down_proj)]:
        target = unsloth_dequantize(layer)
        ours = your_dequantize_nf4(layer, use_custom_asm=False, use_cache_eviction=False, use_optimized=True)
        diff = (target - ours).abs()
        max_diff = diff.max().item()
        mismatches = (diff > 1e-5).sum().item()
        idx = diff.view(-1).argmax().item()
        results[name] = {
            "max_diff": max_diff,
            "mismatches": mismatches,
            "idx": idx,
            "target": target.view(-1)[idx].item(),
            "ours": ours.view(-1)[idx].item(),
        }
    print(results)


@app.function(image=image, gpu="T4", timeout=900)
def debug_forward():
    import os
    import torch
    from transformers import set_seed
    from challenges.challenge_a_nf4 import MLP, mlp_forward, your_dequantize_nf4

    os.chdir("/workspace")
    set_seed(3407)
    hd, m, dt = 2048, 8192, torch.float16
    mlp = MLP(hd=hd, m=m, dtype=dt)
    X = torch.randn((2, 3333, hd), device="cuda", dtype=dt)
    ref = mlp(X)
    ours = mlp_forward(X, mlp, your_dequantize_nf4)
    diff = (ref - ours).abs()
    print({"max_diff": diff.max().item(), "mismatches_gt_1e-5": (diff > 1e-5).sum().item()})
