# SGFT baseline

Solution Guidance Fine-Tuning (arXiv:2412.09906) as used for the SGFT rows of Table 1.

```
teacher (GPT) ──► data/sg_<dataset>.jsonl ──► drpc/student (strategy=sgft, same LoRA recipe as every method)
                                                      │ guide checkpoint
                                              infer/collab_infer.py: guide model -> solution guidance -> response model -> answer
                                                      │
                                              eval/eval_all.py
```

- Solution guidance (SG): a few high-level steps without calculations or a final answer.
- Guide model: the student fine-tuned to write the SG for a question.
- Response model: a model that receives the question and the SG and produces the answer.

## Files

```
data_prep/generate_sg.py   query the teacher for the SG of every training question (zero-shot); requires OPENAI_API_KEY
data_prep/clean_sg.py      normalise and filter SG rows
data/sg_<dataset>.jsonl.gz the released SG training data, rows {id, question, response_rationale}
infer/collab_infer.py      collaborative inference (guide checkpoint + response model) on a test file
sgft_stage2.py             second-stage generation from a saved SG file
eval/eval_all.py           scorer for the collaborative-inference outputs
configs/*.yaml             per-student settings
tests/                     unit tests (pytest)
```

## Commands

```bash
python baselines/sgft/data_prep/generate_sg.py --dataset gsm8k --train-file data/GSM8K/train.jsonl --output-dir baselines/sgft/data
STRATEGY=sgft DATA_FILE=baselines/sgft/data/sg_gsm8k.jsonl TEST_FILE=data/GSM8K/test.jsonl MODEL=Qwen/Qwen3-1.7B RUN=1 sbatch scripts/train_student.slurm
python baselines/sgft/infer/collab_infer.py --guide-ckpt <guide checkpoint> --response-model Qwen/Qwen3-1.7B \
    --dataset gsm8k --test-file data/GSM8K/test.jsonl --output-file outputs/sgft/gsm8k_collab.jsonl
python baselines/sgft/eval/eval_all.py outputs/sgft/gsm8k_collab.jsonl
```

Run every script from the repository root; `--help` lists the remaining options.
