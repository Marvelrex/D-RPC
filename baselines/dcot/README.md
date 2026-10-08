# DCoT baseline

Divergent Chain-of-Thought fine-tuning (Puerto et al., ACL 2025). We used the official
implementation, [UKPLab/acl2025-diverse-cot](https://github.com/UKPLab/acl2025-diverse-cot), through
our fork [Marvelrex/acl2025-diverse-cot](https://github.com/Marvelrex/acl2025-diverse-cot), which adds
the per-dataset LoRA launchers (`*_DCoT_LoRA.slurm`), the direct-prompting launcher
(`DCoT_Baseline_Direct.slurm`) and the data-preparation settings used for the paper.

The students were trained with the same base models, LoRA configuration, learning rate, batch size,
number of epochs and seed as every other method (see the top-level README). Predictions were scored
with `drpc/eval/eval_json_accuracy.py`; the GSM8K/Llama and AQUA/Llama cells use the `correct` field
written by the DCoT evaluation script, as noted in `predictions/table1/README.md`.
