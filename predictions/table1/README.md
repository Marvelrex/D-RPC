# Table-1 predictions

Released per-question student outputs for the Table 1 configurations: 5 datasets x {CoT, Freeform,
SuperCorrect, DCoT, SGFT, RPB (= D-RPC)} x {Llama-3.1-8B-Instruct, Qwen3-1.7B}, with the coverage
listed below. The training protocol uses ten LoRA runs per cell with seed 42. Variation across
training runs comes from GPU non-determinism. Files are gzip JSONL stored with git LFS.

## Original results and released artifacts

The paper's Table 1 reports the original experimental results. Some original artifacts were removed
during cluster cleanup, and the affected experiments were rerun with the same configuration.
The released collection includes these replacement artifacts. The reruns retain the overall
performance trend, while individual summary values can differ from those in the paper.

`paper_table_verification.csv` keeps `paper_mean` and `paper_std` as the paper reference values,
and `ours_mean` and `ours_std` as the recorded available-artifact statistics. `summary.md` reports
statistics from the released prediction files. The per-cell notes below identify the remaining
differences and missing coverage.

```
predictions/table1/<Dataset>/<Method>/<Model>/
    lora/Run01..Run10/predictions.jsonl.gz   fine-tuned student, one run each (the Table-1 cells)
    direct_prompt/predictions.jsonl.gz       un-tuned base model with the same prompt (not in Table 1)
```

Reading a file:

```python
import gzip, json
rows = [json.loads(line) for line in gzip.open(path, "rt")]
```

Score released predictions with `python drpc/eval/eval_json_accuracy.py <file>`, which counts
unparseable outputs as incorrect. `python -m drpc.eval.summarize_table1` applies this scorer to
the released files and prints their means and sample standard deviations (`summary.md`).
The verification report additionally records the baseline's own `correct` field for three cells:
GSM8K/DCoT/Llama, AQUA/DCoT/Llama and AQUA/SGFT/Llama. Their recorded scoring rule is `own`.

`manifest.csv` lists every file with its row count, accuracy, scoring rule and the md5 of the
uncompressed content; `paper_table_verification.csv` compares every cell with the paper value.

The files are verbatim model outputs. Some Qwen3-1.7B students answer partly or entirely in
Chinese (most visibly the DCoT students on AI2ARC); these outputs are kept as generated and scored
by their extracted answer like every other row.

## Coverage: 55 of 56 cells

- 52 cells are marked `EXACT` in the verification report, including small differences of 0.01
  at the displayed precision.
- 3 cells (`INCLUDED_PAPER_MISMATCH`) have complete released files with different statistics:
  - StrategyQA/SGFT, both models: the ten released files are byte-identical, giving a standard
    deviation of 0. The original paper reports 1.29 (Llama) and 2.78 (Qwen), with matching means.
    These released files do not recover the variance of the original runs.
  - GSM8K/SuperCorrect/Qwen3-1.7B: the complete released files give 75.88 +- 0.91, while the paper
    reports 76.44 +- 0.58. D-RPC remains higher than SuperCorrect in this comparison.
- 1 cell (`NOT_INCLUDED`): AI2ARC/CoT/Qwen3-1.7B.
