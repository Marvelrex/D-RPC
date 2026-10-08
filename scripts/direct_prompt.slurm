#!/bin/bash
#SBATCH --job-name=drpc_direct
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/direct_%j.out
#SBATCH --error=logs/direct_%j.err
# Direct prompting of the un-tuned base model (no training) with a prompting strategy.
# Example: DATASET_NAME=gsm8k DATASET_PATH=data/GSM8K/test.jsonl STRATEGY=cot MODEL=Qwen/Qwen3-1.7B sbatch scripts/direct_prompt.slurm
set -euo pipefail
source "${VENV_ACTIVATE:?set VENV_ACTIVATE to your virtualenv activate script}"
cd "${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; mkdir -p logs

DATASET_NAME="${DATASET_NAME:-gsm8k}"
DATASET_PATH="${DATASET_PATH:-data/GSM8K/test.jsonl}"
STRATEGY="${STRATEGY:-cot}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TASK_TYPE="${TASK_TYPE:-math}"
NUM_SAMPLES="${NUM_SAMPLES:-$(wc -l < "$DATASET_PATH")}"
OUT="${OUTPUT_ROOT:-outputs/direct_prompt}/${DATASET_NAME}/${MODEL##*/}/${STRATEGY}"
mkdir -p "$OUT"

python drpc/teacher/query_llama.py \
  --dataset-name "$DATASET_NAME" --dataset-path "$DATASET_PATH" \
  --sample-index 0 --num-samples "$NUM_SAMPLES" \
  --strategy "$STRATEGY" --model "$MODEL" --task-type "$TASK_TYPE" \
  --output-dir "$OUT" --no-do-sample --temperature 0.0 ${DETAILED:+--detailed}

python drpc/eval/eval_json_accuracy.py "$OUT/results_${STRATEGY}.jsonl" --format-validity | tail -1
