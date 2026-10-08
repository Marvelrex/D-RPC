# Data

All five benchmarks used in the paper, with the teacher data that the students were trained on.
Teacher files are gzip-compressed; run `bash scripts/unpack_data.sh` to decompress them in place.
`manifest.csv` lists every file with its row count and the md5 of the uncompressed content.

```
data/<DATASET>/
    <train / test files>              plain JSON / JSONL (see below)
    teacher/rpb_second_round.json.gz  D-RPC teacher data (stage-2 output), one object per training question
    teacher/cot.jsonl.gz              CoT teacher data
    teacher/freeform.jsonl.gz         Freeform teacher data
    teacher/super_correct.jsonl.gz    SuperCorrect teacher data (GSM8K, AQUA, MATH only)
    teacher/sgft.jsonl.gz             SGFT teacher data (MATH only; the other datasets use baselines/sgft/data/)
```

| dataset | train file | test file | test size | D-RPC training tuples |
|---|---|---|---|---|
| GSM8K | `train.jsonl` | `test.jsonl` | 1319 | 10000 |
| AQUA | (teacher files only) | `gsm8k_format_test.jsonl` | 254 | 10000 |
| StrategyQA | `train.json` | `dev_gsm8k_format.jsonl` (`dev.json` is the source) | 229 | 2061 |
| AI2ARC | `gsm8k_format_train.jsonl` | `gsm8k_format_test.jsonl` | 1557 | 6230 |
| MATH | `gsm8k_format_train_10000.jsonl` | `gsm8k_format_test_5000.jsonl` | 5000 | 10000 |

Test rows carry `question` and `answer`; multiple-choice datasets (AQUA, AI2ARC) also carry
`options`, and all but GSM8K carry an `id`.

## D-RPC teacher file (`rpb_second_round.json`)

One object per training question:

| field | meaning |
|---|---|
| `index`, `qid`, `question` | question identifiers and text |
| `gold_answer`, `gold_answer_extracted` | reference answer |
| `reasoning_path_bank` | the route given to the teacher: `category`, `intent`, `difficulty`, `budget`, the retrieved `reasoning_path` (list of step names) and `reasoning_path_options` |
| `rationale` | the teacher's structured rationale: `{step name: "Step1: ... Step2: ..."}`, keys in path order |
| `ans`, `ans_matches_gold` | the teacher's answer and whether it matches the reference |
| `reasoning_path_final` | the route returned by the teacher (with `reasoning_path` when it revised the path) |
| `reasoning_path_source` | parser label: `reasoning_path_bank` when the teacher returned no `reasoning_path` key or the same path, `llm_generated` when it returned a different one |

About 4-5% of the MATH items and 1% of the AQUA items have an empty `rationale` (the teacher
returned only an answer); these items are kept, and all of them are removed by the correctness
filter used for the FILTERED supplementary arm.

## Baseline teacher files (`cot`, `freeform`, `super_correct`, `sgft`)

One JSON object per line with `index`, `question`, `gold`, `prompt_strategy`, `model`,
`response_payload` (raw teacher reply), `response_rationale`, `response_ans` and
`gold_matches_answer`.
