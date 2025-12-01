# ABOUTME: Modal entrypoint for Challenge B FSDP2 + QLoRA training.
# ABOUTME: Runs distributed training on 2x T4 GPUs using accelerate.

import modal
import os

# Build image with all required dependencies
# FSDP2 requires PyTorch >= 2.6.0
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    .pip_install(
        "torch>=2.6.0",
        "accelerate>=1.0.1",
        "bitsandbytes>=0.43.3",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "trl>=0.11.4",
        "datasets",
        "sentencepiece",
        "protobuf",
        "hf_transfer",
        "triton",
    )
    .add_local_dir("challenges", "/workspace/challenges")
    .add_local_file("challenge_b_train.py", "/workspace/challenge_b_train.py")
    .add_local_file("challenge_b_train_with_part_a.py", "/workspace/challenge_b_train_with_part_a.py")
)

app = modal.App("fsdp2-qlora-challenge-b")

# Accelerate config for FSDP1 + QLoRA (known working)
FSDP1_CONFIG = """
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

# Accelerate config for FSDP2 + QLoRA
# Key differences from FSDP1:
# - fsdp_version: 2
# - fsdp_reshard_after_forward instead of fsdp_sharding_strategy
# - No fsdp_backward_prefetch (not supported in FSDP2)
# - fsdp_cpu_ram_efficient_loading: false (causes issues with FSDP2)
FSDP2_CONFIG = """
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_cpu_ram_efficient_loading: false
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

# Default to FSDP1 config (stable)
FSDP_CONFIG = FSDP1_CONFIG


@app.function(
    image=image,
    gpu="T4:2",  # 2x T4 GPUs
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_fsdp_training(max_steps: int = 60, use_fsdp2: bool = False, use_torch_compile: bool = False):
    """Run FSDP + QLoRA distributed training on 2x T4.

    Args:
        max_steps: Number of training steps
        use_fsdp2: If True, use FSDP2 config; otherwise use FSDP1
        use_torch_compile: If True, enable torch.compile for the model
    """
    import subprocess

    os.chdir("/workspace")

    # Select config based on FSDP version
    config_content = FSDP2_CONFIG if use_fsdp2 else FSDP1_CONFIG
    fsdp_version = "FSDP2" if use_fsdp2 else "FSDP1"
    compile_str = " + torch.compile" if use_torch_compile else ""

    # Write accelerate config
    config_path = "/tmp/fsdp_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print("=" * 60)
    print(f"{fsdp_version} + QLoRA{compile_str} Challenge B Training")
    print("=" * 60)
    print(f"Config written to: {config_path}")
    print(f"Config content:\n{config_content}")

    # Check GPU availability
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Launch distributed training with accelerate
    cmd = [
        "accelerate", "launch",
        "--config_file", config_path,
        "/workspace/challenge_b_train.py",
        "--max_steps", str(max_steps),
        "--use_gradient_checkpointing",
        "--use_cpu_offload",
    ]
    if use_torch_compile:
        cmd.append("--use_torch_compile")

    print(f"\nRunning command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )

    return result.returncode


@app.function(
    image=image,
    gpu="T4",  # Single T4 for baseline
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_single_gpu_baseline(max_steps: int = 60):
    """Run single GPU baseline for loss comparison."""
    import subprocess

    os.chdir("/workspace")

    print("=" * 60)
    print("Single GPU Baseline Training")
    print("=" * 60)

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cmd = [
        "python", "/workspace/challenge_b_train.py",
        "--single_gpu_baseline",
        "--max_steps", str(max_steps),
        "--use_gradient_checkpointing",
    ]

    print(f"\nRunning command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )

    return result.returncode


@app.function(
    image=image,
    gpu="T4:2",
    timeout=600,
)
def test_fsdp2_setup():
    """Quick test to verify FSDP2 setup works."""
    import torch
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    print("=" * 60)
    print("Testing FSDP2 Setup")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

    # Test accelerate imports
    try:
        from accelerate import Accelerator
        print("accelerate imported successfully")

        # Check FSDP2 support
        import accelerate
        print(f"accelerate version: {accelerate.__version__}")
    except Exception as e:
        print(f"Error importing accelerate: {e}")

    # Test bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
    except Exception as e:
        print(f"Error importing bitsandbytes: {e}")

    # Test transformers
    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except Exception as e:
        print(f"Error importing transformers: {e}")

    # Test peft
    try:
        import peft
        print(f"peft version: {peft.__version__}")

        # Check if fsdp_auto_wrap_policy exists
        from peft.utils.other import fsdp_auto_wrap_policy
        print("fsdp_auto_wrap_policy available")
    except Exception as e:
        print(f"Error with peft: {e}")

    # Test trl
    try:
        import trl
        print(f"trl version: {trl.__version__}")
    except Exception as e:
        print(f"Error importing trl: {e}")

    return "Setup test completed"


@app.function(
    image=image,
    gpu="T4:2",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_part_a_kernel_training(
    max_steps: int = 60,
    use_part_a: bool = True,
    use_torch_compile: bool = False,
    disable_reacquire: bool = False,
    disable_gather: bool = False,
    disable_scatter: bool = False,
    enable_gather_cache: bool = False,
    enable_sharded_parta: bool = False,
    model_name=None,
    per_device_train_batch_size: int = 2,
    max_seq_length=None,
    tiny_sanity: bool = False,
):
    """Run FSDP2 + QLoRA training with Part A NF4 kernel."""
    import subprocess

    os.chdir("/workspace")

    config_content = FSDP2_CONFIG
    kernel_str = "Part A kernel" if use_part_a else "BnB"
    compile_str = " + torch.compile" if use_torch_compile else ""

    # Write accelerate config
    config_path = "/tmp/fsdp_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print("=" * 60)
    print(f"FSDP2 + QLoRA + {kernel_str}{compile_str} Training")
    print("=" * 60)

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    cmd = [
        "accelerate", "launch",
        "--config_file", config_path,
        "/workspace/challenge_b_train_with_part_a.py",
        "--max_steps", str(max_steps),
        "--use_gradient_checkpointing",
    ]
    if model_name:
        cmd += ["--model_name", model_name]
    if max_seq_length is not None:
        cmd += ["--max_seq_length", str(max_seq_length)]
    cmd += ["--per_device_train_batch_size", str(per_device_train_batch_size)]
    if use_part_a:
        cmd.append("--use_part_a_kernel")
    if use_torch_compile:
        cmd.append("--use_torch_compile")
    if tiny_sanity:
        cmd.append("--tiny_sanity")

    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if disable_reacquire:
        env["ORACLE_PARTA_REACQUIRE"] = "0"
    if disable_gather:
        env["ORACLE_PARTA_GATHER"] = "0"
    if disable_scatter:
        env["ORACLE_PARTA_SCATTER_FALLBACK"] = "0"
    if enable_gather_cache:
        env["ORACLE_PARTA_GATHER_CACHE"] = "1"
    if enable_sharded_parta or use_part_a:
        env["ORACLE_PARTA_SHARDED"] = "1"
    if tiny_sanity:
        env["ORACLE_PARTA_TINY"] = "1"

    if use_part_a:
        print("Part A env toggles:",
              f"reacquire={'off' if disable_reacquire else 'on'}",
              f"gather={'off' if disable_gather else 'on'}",
              f"scatter={'off' if disable_scatter else 'on'}",
              f"gather_cache={'on' if enable_gather_cache else 'off'}",
              f"sharded_parta={'on' if enable_sharded_parta else 'off'}",
              sep=" | ")

    print(f"\nRunning command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        env=env,
    )

    return result.returncode


@app.local_entrypoint()
def main(
    test_only: bool = False,
    baseline: bool = False,
    fsdp2: bool = False,
    compile: bool = False,
    part_a: bool = False,
    max_steps: int = 60,
    model_name=None,
    per_device_train_batch_size: int = 2,
    max_seq_length=None,
    disable_reacquire: bool = False,
    disable_gather: bool = False,
    disable_scatter: bool = False,
    enable_gather_cache: bool = False,
    enable_sharded_parta: bool = False,
):
    """
    Run Challenge B FSDP + QLoRA training.

    Args:
        test_only: Just test the setup without training
        baseline: Run single GPU baseline for comparison
        fsdp2: Use FSDP2 instead of FSDP1
        compile: Enable torch.compile for the model
        part_a: Use Part A NF4 kernel instead of BnB
        max_steps: Number of training steps
        disable_reacquire: Disable Part A reacquire path
        disable_gather: Disable Part A all-gather path
        disable_scatter: Disable Part A scatter fallback path
        enable_gather_cache: Enable caching of all-gathered packed bytes
    """
    if test_only:
        result = test_fsdp2_setup.remote()
        print(result)
    elif baseline:
        returncode = run_single_gpu_baseline.remote(max_steps=max_steps)
        print(f"Baseline training completed with return code: {returncode}")
    elif part_a:
        returncode = run_part_a_kernel_training.remote(
            max_steps=max_steps,
            use_part_a=True,
            use_torch_compile=compile,
            disable_reacquire=disable_reacquire,
            disable_gather=disable_gather,
            disable_scatter=disable_scatter,
            enable_gather_cache=enable_gather_cache,
            enable_sharded_parta=enable_sharded_parta,
            model_name=model_name,
            per_device_train_batch_size=per_device_train_batch_size,
            max_seq_length=max_seq_length,
            tiny_sanity=tiny_sanity,
        )
        compile_str = " + torch.compile" if compile else ""
        print(f"FSDP2 + Part A kernel{compile_str} training completed with return code: {returncode}")
    else:
        returncode = run_fsdp_training.remote(max_steps=max_steps, use_fsdp2=fsdp2, use_torch_compile=compile)
        fsdp_version = "FSDP2" if fsdp2 else "FSDP1"
        compile_str = " + torch.compile" if compile else ""
        print(f"{fsdp_version}{compile_str} training completed with return code: {returncode}")
