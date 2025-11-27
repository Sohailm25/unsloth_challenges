# ABOUTME: Lists OSS issue-solving tasks and scoring rubric.
# ABOUTME: Provides bounty details for Unsloth issues.

r"""
<a name="ISSUES"></a>
## D) Help solve 🦥 Unsloth issues! [Difficulty: Varies] [Max points: 12]

Head over to https://github.com/unslothai/unsloth, and find some issues which are still left standing / not resolved. The tag **currently fixing** might be useful.

Each successfully accepted and solved issue will also have \$100 to \$1000 of bounties.

It's best to attempt these features:

* **<ins>Tool Calling</ins>** [Points = 1] Provide a tool calling Colab notebook and make it work inside of Unsloth. <ins>Bounty: \$1000</ins>

* **<ins>GGUF Vision support</ins>** [Points = 1] Allow exporting vision finetunes to GGUF directly. Llava and Qwen VL must work. <ins>Bounty: \$500</ins>

* **<ins>Refactor Attention</ins>** [Points = 2] Refactor and merge xformers, SDPA, flash-attn, flex-attention into a simpler interface. Must work seamlessly inside of Unsloth. <ins>Bounty: \$350</ins>

* <font color='red'>DONE</font> ** <ins><del>Windows support</del></ins>** [Points = 2] Allow `pip install unsloth` to work in Windows - Triton, Xformers, bitsandbytes should all function. You might need to edit `pyproject.toml`. Confirm it works. <ins>Bounty: \$300</ins>

* **<ins>Support Sequence Classification</ins>** [Points = 1] Create patching functions to patch over AutoModelForSequenceClassification, and allow finetuner to use AutoModelForSequenceClassification. <ins>Bounty: \$200</ins>

* **<ins>VLMs Data Collator</ins>** [Points = 1] Make text & image mixing work efficiently -so some inputs can be text only. Must work on Qwen, Llama, Pixtral. <ins>Bounty: \$100</ins>

* <font color='red'>DONE</font> **<ins>VLMs image resizing</ins>** [Points = 1] Allow finetuner to specify maximum image size, or get it from the config.json file. Resize all images to specific size to reduce VRAM. <ins>Bounty: \$100</ins>

* **<ins>Support Flex Attention</ins>** [Points = 2] Allow dynamic sequence lengths without excessive recompilation. Make this work on SWAs and normal causal masks. Also packed sequence masks. <ins>Bounty: \$100</ins>

* <font color='red'>DONE</font> **<ins>VLMs train only on completions</ins>** [Points = 1] Edit `train_on_responses_only` to allow it to work on VLMs. <ins>Bounty: \$100</ins>

## Marking Criteria for D) Max points = 12
```python
if attemped_D:
    D_score = 0
    for subtask in subtasks:
        if sucessfully_completed_subtask:
            D_score += score_for_subtask
    final_score += D_score
```

---
---
---
"""
