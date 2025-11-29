# ABOUTME: Research notes for Challenge B - QLoRA with FSDP2
# ABOUTME: Contains technical findings, implementation approaches, and scoring strategy

## Description
Research gathered on 2025-11-28 for implementing Challenge B: Making QLoRA work with FSDP2 on 2x Tesla T4 GPUs in Kaggle.

---

## Challenge Requirements Summary

From `challenges/challenge_b_fsdp2.py`:

1. **Goal**: Single Python script to finetune Llama 3.1 8B on 2+ GPUs with FSDP2
2. **Environment**: Free Kaggle notebook with 2x Tesla T4 GPUs
3. **Alternative**: Pipeline parallelism with zero bubble scheduling
4. **Model Source**: Pre-quantized 4bit BnB safetensor from Unsloth's HF page OR full 16bit
5. **Framework**: Can use `accelerate` but must be FSDP2 or related
6. **Compatibility**: Must use `TrainingArguments`/`Trainer` or TRL classes
7. **Loss Equivalence**: Must match single GPU training loss
8. **Features**: Enable all FSDP2 features - offloading, checkpointing, mixed precision
9. **Quantization**: Can use nf4 from torch AO, but best from bitsandbytes
10. **Deliverable**: Working Kaggle 2x T4 notebook

---

## Scoring Criteria (Max 10 points)

```python
if attemped_B:
    B_score = 0
    if FSDP2_works_with_QLoRA:
        if torch_compile_works: B_score += 5
        else: B_score += 3
        if uses_part_A_and_single_kernel_and_faster: B_score += 3
        elif uses_torchAO:
            if torchAO_slower_than_BnB: B_score -= 3
    elif TP_or_PP_with_QLoRA:
        if zero_bubble: B_score += 3
        else: B_score += 2
    elif FSDP1_works_with_QLoRA:
        B_score += 1
    if kaggle_notebook_2_tesla_t4_example:
        B_score += 2
    else:
        B_score = 0  # CRITICAL: No Kaggle demo = 0 points
```

### Optimal Path Analysis

| Approach | Base | torch.compile | Part A Kernel | Kaggle | Total |
|----------|------|---------------|---------------|--------|-------|
| FSDP2 + QLoRA + compile + Part A | 5 | ✓ | +3 | +2 | **10** |
| FSDP2 + QLoRA + compile | 5 | ✓ | 0 | +2 | **7** |
| FSDP2 + QLoRA (no compile) | 3 | - | 0 | +2 | **5** |
| PP + zero bubble + QLoRA | 3 | - | - | +2 | **5** |
| PP + QLoRA | 2 | - | - | +2 | **4** |
| FSDP1 + QLoRA | 1 | - | - | +2 | **3** |

**Target**: FSDP2 + QLoRA + torch.compile + Part A kernel = 10 points

---

## Technical Deep Dive

### 1. FSDP2 Overview

FSDP2 (PyTorch's Fully Sharded Data Parallel v2) improvements over FSDP1:
- **DTensor-based**: Uses per-parameter sharding via DTensor
- **Memory**: Lower, deterministic GPU memory usage
- **Performance**: 12% tokens/sec speedup, 3.2x model init speedup over FSDP1
- **Extension Point**: Tensor subclass for custom all-gather (float8, NF4)

Source: [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### 2. QLoRA + FSDP Challenge

**Core Problem**: FSDP only supports float dtypes, but 4-bit quantized weights use integer storage.

**Solution**: `bnb_4bit_quant_storage` parameter in BitsAndBytesConfig

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,  # KEY: Store as float for FSDP
)
```

**Critical Requirement**: `quant_storage` dtype MUST match model's `torch_dtype`

Source: [bitsandbytes FSDP-QLoRA docs](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/fsdp_qlora.md)

### 3. Accelerate FSDP2 Support

PR #3394 (merged March 2025) adds `fsdp_version=2`:

```python
fsdp_plugin = FullyShardedDataParallelPlugin(fsdp_version=2)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

Or via config:
```yaml
distributed_type: FSDP
fsdp_version: 2
fsdp_sharding_strategy: FULL_SHARD
fsdp_offload_params: true
fsdp_cpu_ram_efficient_loading: true
fsdp_use_orig_params: false  # Required for QLoRA
```

Source: [accelerate PR #3394](https://github.com/huggingface/accelerate/pull/3394)

### 4. FSDP Wrapping Policy for QLoRA

**Problem**: With `use_orig_params=False`, trainable and non-trainable params must be wrapped separately.

**Solution**:
```python
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
```

Source: [PEFT FSDP docs](https://huggingface.co/docs/peft/en/accelerate/fsdp)

### 5. Required Versions

```bash
pip install bitsandbytes>=0.43.3
pip install accelerate>=1.0.1
pip install transformers>4.44.2
pip install trl>0.11.4
pip install peft>0.13.0
```

### 6. Kaggle Multi-GPU Setup

Kaggle 2x T4:
- Single node, world_size=2
- GPU ranks: 0 and 1
- Use `accelerate launch` or `torchrun`

```python
# Kaggle GPU detection
import torch
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
```

Source: [PyTorch DDP Kaggle tutorial](https://learnopencv.com/distributed-parallel-training-pytorch-multi-gpu-setup/)

---

## Implementation Approaches

### Approach A: Accelerate + TRL (Recommended for Transformers compatibility)

```python
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. FSDP2 plugin setup
fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    sharding_strategy="FULL_SHARD",
    cpu_offload=True,
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# 2. BnB config with quant_storage for FSDP compatibility
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# 3. Load model
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# 4. LoRA config
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0, bias="none",
)
model = get_peft_model(model, lora_config)

# 5. Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(...),
)
# Apply FSDP auto wrap policy
from peft.utils.other import fsdp_auto_wrap_policy
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

trainer.train()
```

### Approach B: TorchTune (Native FSDP2)

TorchTune has native FSDP2 + QLoRA recipes:
```bash
tune run --nproc_per_node 2 lora_finetune_fsdp2 --config llama3/8B_qlora
```

But may not satisfy "TrainingArguments/Trainer" requirement.

### Approach C: Axolotl Config

```yaml
adapter: qlora
qlora_sharded_model_loading: true
fsdp_version: 2
fsdp_config:
  offload_params: true
  cpu_offload_pin_memory: false
  cpu_ram_efficient_loading: true
```

---

## Existing Kaggle Notebooks

1. **[FSDP2_QLoRA by yyyu54](https://www.kaggle.com/code/yyyu54/fsdp2-qlora)** - March 2025
2. **[FSDP2 with qlora by mxksowie](https://www.kaggle.com/code/mxksowie/fsdp2-with-qlora)** - April 2025

These can serve as references/starting points.

---

## Integration with Challenge A (Part A Kernel)

For +3 bonus points, integrate the NF4 Triton kernel from Challenge A:
1. Replace bitsandbytes dequantization with `your_dequantize_nf4`
2. Must be faster than BnB's implementation
3. Kernel must work with FSDP2's sharded parameters

**Challenge**: FSDP2 shards parameters across GPUs, so dequantization happens on sharded tensors.

---

## torch.compile Considerations

For the +5 points (vs +3 without compile):
- torch.compile must not cause graph breaks
- QLoRA + FSDP2 + compile is complex
- Reference: Challenge C covers torch.compile specifics

---

## Tesla T4 Constraints

- Compute capability: sm_75
- No native BF16 support (emulated via FP32)
- 16GB VRAM per GPU
- Use FP16 or emulated BF16

---

## Next Steps

1. [ ] Set up Kaggle notebook environment
2. [ ] Implement basic FSDP2 + QLoRA with accelerate
3. [ ] Verify loss equivalence with single-GPU baseline
4. [ ] Enable offloading, checkpointing, mixed precision
5. [ ] Add torch.compile support
6. [ ] Integrate Part A kernel for +3 bonus
7. [ ] Benchmark and document

---

## Sources

- [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [bitsandbytes FSDP-QLoRA](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/fsdp_qlora.md)
- [HuggingFace FSDP-QLoRA Guide](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora)
- [accelerate FSDP2 PR #3394](https://github.com/huggingface/accelerate/pull/3394)
- [PEFT FSDP docs](https://huggingface.co/docs/peft/en/accelerate/fsdp)
- [Answer.AI FSDP-QLoRA](https://github.com/AnswerDotAI/fsdp_qlora)
- [TorchTune FSDP2 QLoRA](https://github.com/meta-pytorch/torchtune)
- [Axolotl FSDP QLoRA](https://axolotl-ai-cloud.github.io/axolotl/docs/fsdp_qlora.html)
