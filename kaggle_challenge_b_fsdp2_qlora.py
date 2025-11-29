# ABOUTME: Kaggle-ready script for FSDP2 + QLoRA + torch.compile training.
# ABOUTME: Run via accelerate on 2x T4 GPUs in Kaggle notebook environment.

"""
FSDP2 + QLoRA + torch.compile Training Script for Kaggle 2x T4

This script is designed to run in a Kaggle notebook with 2x T4 GPUs.
Usage in Kaggle notebook:

    # Cell 1: Install dependencies
    !pip install -q accelerate>=1.0.1 bitsandbytes>=0.43.3 transformers>=4.45.0 peft>=0.13.0 trl>=0.11.4 datasets sentencepiece hf_transfer

    # Cell 2: Write this script to a file
    %%writefile fsdp2_qlora_train.py
    <paste this entire script>

    # Cell 3: Write accelerate config
    %%writefile /root/.cache/huggingface/accelerate/default_config.yaml
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

    # Cell 4: Launch distributed training
    !accelerate launch fsdp2_qlora_train.py --max_steps 60

Requirements:
- Kaggle GPU T4 x2 accelerator
- accelerate >= 1.0.1
- bitsandbytes >= 0.43.3
- transformers >= 4.45.0
- peft >= 0.13.0
- trl >= 0.11.4
"""

import os
import sys
import torch
import argparse
from dataclasses import dataclass

# Set environment variables before imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
)


def parse_args():
    parser = argparse.ArgumentParser(description="FSDP2 + QLoRA Training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=60,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_fsdp2",
        help="Output directory",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        default=True,
        help="Enable torch.compile via TrainingArguments",
    )
    return parser.parse_args()


def get_bnb_config(compute_dtype=torch.float16):
    """Create BitsAndBytesConfig with quant_storage for FSDP compatibility."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        # KEY: quant_storage must be float dtype for FSDP sharding
        bnb_4bit_quant_storage=compute_dtype,
    )


def get_dataset(tokenizer):
    """Load and prepare the training dataset."""
    from datasets import load_dataset

    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")
    return dataset


def run_fsdp2_training(args):
    """Run FSDP2 + QLoRA distributed training."""
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig

    compile_str = "+ torch.compile" if args.use_torch_compile else ""
    print("=" * 60)
    print(f"Running FSDP2 + QLoRA {compile_str} DISTRIBUTED TRAINING")
    print("=" * 60)

    # Initialize accelerator (FSDP2 config comes from accelerate config)
    accelerator = Accelerator()

    is_main = accelerator.is_main_process
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes

    if is_main:
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {accelerator.device}")

    # T4 uses FP16 (no native BF16)
    compute_dtype = torch.float16

    bnb_config = get_bnb_config(compute_dtype)

    if is_main:
        print(f"Loading model: {args.model_name}")

    # Load model - don't use device_map with FSDP
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
        # Note: device_map should NOT be used with FSDP
        # FSDP handles device placement
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable input require grads for LoRA
    model.enable_input_require_grads()

    dataset = get_dataset(tokenizer)

    # LoRA config - passed to SFTTrainer instead of pre-wrapping
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build training args
    training_args_kwargs = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=1,
        max_steps=args.max_steps,
        logging_steps=1,
        output_dir=args.output_dir,
        seed=3407,
        dataset_text_field="text",
        fp16=True,
        bf16=False,
        report_to="none",
        dataset_num_proc=4,
        save_strategy="no",
        gradient_checkpointing=args.use_gradient_checkpointing,
        # Use non-fused AdamW to avoid FSDP broadcast shape issues
        optim="adamw_torch",
    )

    # Add torch_compile via TrainingArguments if requested
    # This is the correct way to enable compile with FSDP2 + QLoRA
    if args.use_torch_compile:
        if is_main:
            print("Enabling torch.compile via TrainingArguments...")
        training_args_kwargs["torch_compile"] = True
        training_args_kwargs["torch_compile_mode"] = "default"

    training_args = SFTConfig(**training_args_kwargs)

    # Pass peft_config to SFTTrainer - it handles LoRA application with FSDP
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    if is_main:
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    if is_main:
        print("Starting FSDP2 training...")

    train_result = trainer.train()

    if is_main:
        print("\n" + "=" * 60)
        print("FSDP2 TRAINING RESULTS")
        print("=" * 60)
        print(f"Final loss: {train_result.training_loss:.4f}")

        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        print(f"Loss curve (first 10): {losses[:10]}")
        print(f"Loss curve (last 10): {losses[-10:]}")

    return train_result


def main():
    args = parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    run_fsdp2_training(args)


if __name__ == "__main__":
    main()
