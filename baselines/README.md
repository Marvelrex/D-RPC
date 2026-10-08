# Baselines

Every baseline student is trained and scored with the same recipe and scorer as D-RPC
(`scripts/train_student.slurm`, `drpc/eval/eval_json_accuracy.py`); only the teacher data differs.

| baseline | original work | implementation used for the paper |
|---|---|---|
| CoT | zero-shot chain-of-thought distillation | this repository: strategy `cot` (`drpc/teacher/prompts.py`, `drpc/student/distill_rationale.py`) |
| Freeform | D-RPC output format without any route or path (bank ablation) | this repository: strategy `freeform` |
| SuperCorrect | [YangLing0818/SuperCorrect-llm](https://github.com/YangLing0818/SuperCorrect-llm) (Yang et al.) | this repository: strategy `super_correct` re-implements the hierarchical thought-template teacher prompt and trains the student with the shared recipe |
| SGFT | [BiJings/SGFT](https://github.com/BiJings/SGFT) (Bi et al., COLING 2025) | this repository: [`sgft/`](sgft/README.md), our re-implementation of solution-guidance generation, guide-model fine-tuning (strategy `sgft`) and collaborative inference |
| DCoT | [UKPLab/acl2025-diverse-cot](https://github.com/UKPLab/acl2025-diverse-cot) (Puerto et al., ACL 2025) | the official code, run from our fork [Marvelrex/acl2025-diverse-cot](https://github.com/Marvelrex/acl2025-diverse-cot), see [`dcot/`](dcot/README.md) |
| direct prompting | un-tuned base model with the same prompt | this repository: `drpc/teacher/query_llama.py`, `scripts/direct_prompt.slurm` |
