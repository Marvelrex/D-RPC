# Table-1 predictions

Per-question student outputs behind Table 1 of the paper: 5 datasets x {CoT, Freeform, SuperCorrect,
DCoT, SGFT, RPB (= D-RPC)} x {Llama-3.1-8B-Instruct, Qwen3-1.7B}, 10 LoRA runs per cell
(seed 42 for every run; run-to-run variation comes from GPU non-determinism). Files are gzip
JSONL stored with git LFS.

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

Scoring that reproduces the paper: `python drpc/eval/eval_json_accuracy.py <file>` (unparseable
outputs count as incorrect); `python -m drpc.eval.summarize_table1` rebuilds the whole table
(`summary.md`). Three DCoT/SGFT cells (GSM8K/DCoT/Llama, AQUA/DCoT/Llama, AQUA/SGFT/Llama)
match the paper only when scored with the `correct` field written by the DCoT evaluation script.

`manifest.csv` lists every file with its row count, accuracy, scoring rule and the md5 of the
uncompressed content; `paper_table_verification.csv` compares every cell with the paper value.

The files are verbatim model outputs. Some Qwen3-1.7B students answer partly or entirely in
Chinese (most visibly the DCoT students on AI2ARC); these outputs are kept as generated and scored
by their extracted answer like every other row.

## Coverage: 55 of 56 cells

- 52 cells (`EXACT`) reproduce the paper's mean and standard deviation.
- 3 cells (`INCLUDED_PAPER_MISMATCH`) are complete but do not reproduce the paper number:
  - StrategyQA/SGFT, both models: the 10 runs are byte-identical (identical adapters), so the
    standard deviation is 0 where the paper reports 1.29 (Llama) and 2.78 (Qwen); the means match.
  - GSM8K/SuperCorrect/Qwen3-1.7B: the paper's per-run accuracies were computed on partially
    generated files that were later completed; the complete files give 75.88 +- 0.91 instead of
    76.44 +- 0.58. The ranking in Table 1 is unchanged.
- 1 cell (`NOT_INCLUDED`): AI2ARC/CoT/Qwen3-1.7B.
