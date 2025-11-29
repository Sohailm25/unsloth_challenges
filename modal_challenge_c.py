# ABOUTME: Modal entrypoint to run torch.compile QLoRA challenge tests on GPU.
# ABOUTME: Builds GPU image with PyTorch 2.7+ for flex_attention + dynamic shapes support.

import modal
import sys

# Use PyTorch 2.7+ for flex_attention with dynamic shapes support
# PyTorch 2.5.1/2.6.0 have known issues with flex_attention + dynamic=True
# See: research/oracle_flex_attention_dynamic_shapes.md
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04")
    .apt_install("build-essential", "git", "python3", "python3-pip", "python3-venv")
    .env({"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"})  # Set PATH before run_commands
    .run_commands(
        # Create python symlink for Modal's pip_install
        "ln -sf /usr/bin/python3 /usr/bin/python",
        # Create venv to avoid PEP 668 issues with system python on Ubuntu 24.04
        "python3 -m venv /opt/venv",
        # Install PyTorch 2.7+ with CUDA 12.6 support in venv
        "/opt/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",
        # Install ML packages
        "/opt/venv/bin/pip install transformers>=4.46.0 peft>=0.13.0 trl>=0.12.0 bitsandbytes>=0.44.0",
        "/opt/venv/bin/pip install datasets accelerate sentencepiece protobuf huggingface_hub hf_transfer pytest",
        # Symlink venv python as default
        "ln -sf /opt/venv/bin/python /usr/local/bin/python",
        "ln -sf /opt/venv/bin/python3 /usr/local/bin/python3",
    )
    .add_local_dir("challenges", "/workspace/challenges")
    .add_local_file("modal_challenge_c.py", "/workspace/modal_challenge_c.py")
)

app = modal.App("torch-compile-challenge")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,  # 30 min for compilation
)
def run_training():
    """Run the torch.compile QLoRA training script."""
    import os
    import sys

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    # Import and run the solution
    from challenges.challenge_c_solution import main
    main()


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
)
def run_graph_break_test():
    """Test for graph breaks in the compiled model."""
    import os
    import sys
    import torch

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    # Configure logging to capture graph breaks
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"

    from challenges.challenge_c_solution import (
        apply_all_patches,
        setup_torch_compile_logging,
        prepare_peft_for_compile,
        HAS_FLEX_ATTENTION,
        NF4_CUSTOM_OP_AVAILABLE,
    )

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"flex_attention available: {HAS_FLEX_ATTENTION}")
    print(f"NF4 custom op available: {NF4_CUSTOM_OP_AVAILABLE}")

    setup_torch_compile_logging()
    # Enable Part A kernel for +1 scoring points
    apply_all_patches(use_flex_attention=HAS_FLEX_ATTENTION, use_part_a_kernel=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    # Load model with patches applied
    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        torch_dtype=dtype,  # Force fp16 for non-quantized layers (T4 compat)
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # CRITICAL: Convert PEFT LoRA scaling floats to tensor buffers for torch.compile
    # This allows PEFT forward to stay in-graph without dynamo.disable (-2 penalty)
    prepare_peft_for_compile(model)

    model.enable_input_require_grads()

    # Test forward pass
    print("\n[Test] Running forward pass...")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Forward pass successful. Output shape: {outputs.logits.shape}")

    # Test with different sequence lengths to check recompilation
    print("\n[Test] Testing different sequence lengths...")
    test_sequences = [
        "Hi",
        "Hello, how are you today?",
        "This is a much longer sequence to test dynamic shapes in the compiled model.",
    ]

    for seq in test_sequences:
        inputs = tokenizer(seq, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"  Seq len {len(seq):3d}: output shape {outputs.logits.shape}")

    print("\n[Test] Graph break test completed!")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
)
def run_loss_comparison():
    """Compare loss between compiled and non-compiled models."""
    import os
    import sys
    import torch

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    # Load non-compiled model first
    print("[Test] Loading non-compiled model...")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model_baseline = get_peft_model(model_baseline, lora_config)

    # Compute baseline loss
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    labels = inputs["input_ids"].clone()

    with torch.no_grad():
        outputs_baseline = model_baseline(**inputs, labels=labels)
        loss_baseline = outputs_baseline.loss
        print(f"Baseline loss: {loss_baseline.item():.6f}")

    # Clear memory
    del model_baseline
    torch.cuda.empty_cache()

    # Now load compiled model
    print("\n[Test] Loading compiled model...")
    from challenges.challenge_c_solution import apply_all_patches, HAS_FLEX_ATTENTION, NF4_CUSTOM_OP_AVAILABLE
    print(f"NF4 custom op available: {NF4_CUSTOM_OP_AVAILABLE}")
    apply_all_patches(use_flex_attention=HAS_FLEX_ATTENTION, use_part_a_kernel=True)

    model_compiled = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        quantization_config=bnb_config,
    )
    model_compiled = get_peft_model(model_compiled, lora_config)

    with torch.no_grad():
        outputs_compiled = model_compiled(**inputs, labels=labels)
        loss_compiled = outputs_compiled.loss
        print(f"Compiled loss: {loss_compiled.item():.6f}")

    # Compare
    diff = abs(loss_baseline.item() - loss_compiled.item())
    rel_diff = diff / loss_baseline.item() * 100
    print(f"\nLoss difference: {diff:.6f} ({rel_diff:.4f}%)")
    # 1% tolerance for numerical precision differences in compiled code
    print(f"Losses match (within 1%): {rel_diff < 1.0}")


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def check_environment():
    """Check the environment and available features."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check flex_attention
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        print("flex_attention: AVAILABLE")
    except ImportError as e:
        print(f"flex_attention: NOT AVAILABLE ({e})")

    # Check compiled_autograd
    try:
        import torch._dynamo
        print(f"torch._dynamo available: True")
        print(f"compiled_autograd config available: {hasattr(torch._dynamo.config, 'compiled_autograd')}")
    except Exception as e:
        print(f"torch._dynamo error: {e}")

    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("bitsandbytes: NOT AVAILABLE")

    # Check transformers
    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except ImportError:
        print("transformers: NOT AVAILABLE")


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
)
def run_training_a100():
    """Run training on A100 with flex_attention enabled."""
    import os
    import sys

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    from challenges.challenge_c_solution import main
    # Run with Part A kernel (custom_op) enabled for scoring points
    # custom_op avoids dynamo.disable penalty and stays in-graph
    main(use_part_a_kernel=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
)
def check_environment_a100():
    """Check the A100 environment and flex_attention support."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU compute capability: sm{major}{minor}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Supports flex_attention+max_autotune: {major >= 8}")

    # Check flex_attention
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        print("flex_attention: AVAILABLE")
    except ImportError as e:
        print(f"flex_attention: NOT AVAILABLE ({e})")

    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("bitsandbytes: NOT AVAILABLE")

    # Check transformers
    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except ImportError:
        print("transformers: NOT AVAILABLE")


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
)
def run_graph_break_test_a100():
    """Test for graph breaks on A100 with flex_attention."""
    import os
    import sys
    import torch

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"

    from challenges.challenge_c_solution import (
        apply_all_patches,
        setup_torch_compile_logging,
        should_enable_flex_attention,
        get_gpu_capability,
        prepare_peft_for_compile,
        NF4_CUSTOM_OP_AVAILABLE,
    )

    major, minor = get_gpu_capability()
    can_use_flex = should_enable_flex_attention()
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm{major}{minor})")
    print(f"should_enable_flex_attention(): {can_use_flex}")
    print(f"NF4 custom op available: {NF4_CUSTOM_OP_AVAILABLE}")

    setup_torch_compile_logging()
    # Enable Part A kernel for +1 scoring points
    apply_all_patches(use_flex_attention=True, use_part_a_kernel=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        torch_dtype=dtype,  # Force fp16 for non-quantized layers (T4 compat)
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # CRITICAL: Convert PEFT LoRA scaling floats to tensor buffers for torch.compile
    # This allows PEFT forward to stay in-graph without dynamo.disable (-2 penalty)
    prepare_peft_for_compile(model)

    model.enable_input_require_grads()

    print("\n[Test] Running forward pass with flex_attention on A100...")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Forward pass successful. Output shape: {outputs.logits.shape}")

    print("\n[Test] Testing different sequence lengths...")
    test_sequences = [
        "Hi",
        "Hello, how are you today?",
        "This is a much longer sequence to test dynamic shapes with flex_attention.",
    ]

    for seq in test_sequences:
        inputs = tokenizer(seq, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"  Seq len {len(seq):3d}: output shape {outputs.logits.shape}")

    print("\n[Test] Graph break test on A100 completed!")


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
)
def debug_flex_attention():
    """Debug flex_attention vs SDPA to find numerical issues."""
    import os
    import sys
    import math

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test flex_attention vs SDPA on synthetic data
    torch.manual_seed(42)

    bsz, num_heads, seq_len, head_dim = 1, 32, 64, 64
    dtype = torch.float16

    Q = torch.randn(bsz, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    K = torch.randn(bsz, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    V = torch.randn(bsz, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    scale = 1.0 / math.sqrt(head_dim)

    print("\n=== Test 1: flex_attention vs SDPA (causal) ===")

    # SDPA reference
    sdpa_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)
    print(f"SDPA output: min={sdpa_out.min():.4f}, max={sdpa_out.max():.4f}, mean={sdpa_out.mean():.4f}")
    print(f"SDPA has nan: {sdpa_out.isnan().any()}, has inf: {sdpa_out.isinf().any()}")

    # flex_attention with block mask
    def causal_mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mask_fn,
        B=bsz,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device="cuda",
    )

    flex_out = flex_attention(Q, K, V, block_mask=block_mask, scale=scale)
    print(f"flex_attention output: min={flex_out.min():.4f}, max={flex_out.max():.4f}, mean={flex_out.mean():.4f}")
    print(f"flex_attention has nan: {flex_out.isnan().any()}, has inf: {flex_out.isinf().any()}")

    # Compare
    diff = (sdpa_out - flex_out).abs()
    print(f"Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
    close = torch.allclose(sdpa_out, flex_out, rtol=1e-2, atol=1e-2)
    print(f"Outputs match (rtol=1e-2, atol=1e-2): {close}")

    print("\n=== Test 2: Full model forward pass comparison ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    # Load model WITHOUT patches (baseline)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("\nLoading baseline model (SDPA)...")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=bnb_config,
    )
    model_baseline = get_peft_model(model_baseline, lora_config)

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    labels = inputs["input_ids"].clone()

    with torch.no_grad():
        out_baseline = model_baseline(**inputs, labels=labels)
        loss_baseline = out_baseline.loss
        logits_baseline = out_baseline.logits
        print(f"Baseline loss: {loss_baseline.item():.4f}")
        print(f"Baseline logits: min={logits_baseline.min():.4f}, max={logits_baseline.max():.4f}")
        print(f"Baseline logits nan: {logits_baseline.isnan().any()}, inf: {logits_baseline.isinf().any()}")

    del model_baseline
    torch.cuda.empty_cache()

    print("\nLoading flex_attention model (with patches)...")
    from challenges.challenge_c_solution import apply_all_patches, prepare_peft_for_compile

    # Apply patches for flex_attention
    apply_all_patches(use_flex_attention=True, use_part_a_kernel=True)

    model_flex = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        quantization_config=bnb_config,
    )
    model_flex = get_peft_model(model_flex, lora_config)
    prepare_peft_for_compile(model_flex)

    with torch.no_grad():
        out_flex = model_flex(**inputs, labels=labels)
        loss_flex = out_flex.loss
        logits_flex = out_flex.logits
        print(f"flex_attention loss: {loss_flex.item():.4f}")
        print(f"flex_attention logits: min={logits_flex.min():.4f}, max={logits_flex.max():.4f}")
        print(f"flex_attention logits nan: {logits_flex.isnan().any()}, inf: {logits_flex.isinf().any()}")

    print("\n=== Comparison ===")
    loss_diff = abs(loss_baseline.item() - loss_flex.item())
    print(f"Loss difference: {loss_diff:.6f} ({loss_diff/loss_baseline.item()*100:.2f}%)")

    logits_diff = (logits_baseline - logits_flex).abs()
    print(f"Logits difference: max={logits_diff.max():.4f}, mean={logits_diff.mean():.4f}")

    if loss_diff > 0.1 or logits_flex.isnan().any():
        print("\n*** WARNING: Significant numerical differences detected! ***")
    else:
        print("\n*** Outputs are numerically similar ***")


@app.local_entrypoint()
def main(action: str = "check"):
    """
    Run Challenge C tests on Modal.

    Actions:
    - check: Check environment (T4)
    - check-a100: Check A100 environment
    - train: Run training (T4)
    - train-a100: Run training on A100 with flex_attention
    - graph: Test for graph breaks (T4)
    - graph-a100: Test graph breaks on A100 with flex_attention
    - loss: Compare compiled vs non-compiled loss
    - debug-flex: Debug flex_attention vs SDPA on A100
    """
    if action == "check":
        check_environment.remote()
    elif action == "check-a100":
        check_environment_a100.remote()
    elif action == "train":
        run_training.remote()
    elif action == "train-a100":
        run_training_a100.remote()
    elif action == "graph":
        run_graph_break_test.remote()
    elif action == "graph-a100":
        run_graph_break_test_a100.remote()
    elif action == "loss":
        run_loss_comparison.remote()
    elif action == "debug-flex":
        debug_flex_attention.remote()
    else:
        print(f"Unknown action: {action}")
        print("Available actions: check, check-a100, train, train-a100, graph, graph-a100, loss, debug-flex")
