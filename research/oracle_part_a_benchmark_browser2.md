# Oracle: Part A FSDP2 performance guidance (browser retry)

- Date: 2025-11-29
- Engine: browser (gpt-5.1-pro)
- Prompt: "How to make Part A NF4 kernel faster and numerically aligned than BnB under FSDP2 on 2x T4; gather path currently 266s vs BnB 162s and loss ~14 vs 6.9. Want actionable tweaks (env toggles, shard handling, kernel args) using challenge_b_train_with_part_a.py."
- Files: reference/full_prd.md, challenge_b_train_with_part_a.py, runs/benchmarks/2025-11-29-part-a.log
- Outcome: Failed (ChatGPT session not detected in browser automation; login button present). No guidance returned. Did not use API key per instruction.
