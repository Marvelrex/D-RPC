# Supplementary: reasoning-path bank ablation

Does the content of the retrieved reasoning path matter? Four variants of the D-RPC teacher data
are built for MATH and AQUA, students are trained with the unchanged recipe, and accuracy is compared
with the full D-RPC students of Table 1.

| arm | teacher input | training tuples (MATH / AQUA) |
|---|---|---|
| `empty` | route without any candidate path (`reasoning_path_options: []`); the teacher writes its own path | 10000 / 10000 |
| `random` | the retrieved path is replaced by a path from an item with a different (category, intent), truncated to the budget | 10000 / 10000 |
| `randglobal` | the retrieved path is replaced by a uniform draw from the whole bank, excluding the item's own retrieved paths | 10000 / 10000 |
| `filtered` | full D-RPC data without the items whose teacher answer is wrong (`ans_matches_gold == false`) | 7482 / 8827 |

The teacher followed the injected path exactly in 41% (MATH) and 52% (AQUA) of the `random` /
`randglobal` items and discarded it otherwise, so those two arms are partial controls. In the data
`random` and `randglobal` are two independent draws of the same construction (category match with the
question 14-15%, chance level 16%; intent match 0-2%).

## Layout

```
data/<MATH|AQUA>/<arm>.json.gz            teacher data per arm (same schema as data/*/teacher/rpb_second_round.json)
predictions/seed42/<DS>/<arm>/<Model>/RunNN/predictions.jsonl.gz
                                           10 runs per cell, seed 42 for every run (the paper protocol);
                                           MATH also has randglobal_fir, a second complete set of the same
                                           arm trained on another cluster
predictions/seeds43-52/AQUA/<arm>/<Model>/RunNN/predictions.jsonl.gz
                                           robustness check: AQUA re-trained with seed = 42 + run
predictions/manifest.csv                   rows, md5 and provenance of every prediction file
results/seed42.md                          per-cell and per-run accuracies, gaps with 95% CI (both datasets)
results/seeds43-52.md                      the same for the AQUA multi-seed check
results/path_stats.md                      path reuse rate and entropy of every teacher file
```

Students: `scripts/train_student.slurm` recipe (LoRA r=64 alpha=128, lr 1e-4, batch 2x8, 2 epochs,
greedy decoding, 2048 new tokens), launched per cell by `run_ablation.slurm` / `keeper.sh`.
Scoring: `drpc/eval/eval_json_accuracy.py`; `analyze.py --campaign seed42` recomputes the reports
from the prediction files, `path_stats.py` the reuse/entropy table.

## Rebuilding the arm data

`make_filtered.py` reproduces `filtered.json` exactly. `perturb_routes.py` is a reconstruction of
the generator that produced `empty`, `random` and `randglobal` (the original script was not kept):
it reproduces the stored teacher prompts verbatim and implements the recipes above, but a new run
draws new random paths and new teacher replies, so it regenerates the construction, not the files.

```bash
bash scripts/unpack_data.sh
python supplementary/bank_ablation/make_filtered.py --source data/MATH/teacher/rpb_second_round.json --output /tmp/filtered.json
python supplementary/bank_ablation/perturb_routes.py --source data/MATH/teacher/rpb_second_round.json --mode randglobal --answer-type numeric --output /tmp/randglobal.json
```
