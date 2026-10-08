#!/bin/bash
# Stage 0: query the teacher once per training question with a prompting strategy.
# STRATEGY in {rpb, cot, freeform, super_correct, sgft, normal}; writes results_<strategy>.jsonl.
# Requires OPENAI_API_KEY. Run from the repository root.
set -euo pipefail
DATASET_NAME="${DATASET_NAME:-gsm8k}"
DATASET_PATH="${DATASET_PATH:-data/GSM8K/train.jsonl}"
STRATEGY="${STRATEGY:-rpb}"
MODEL="${MODEL:-gpt-5.1}"
TASK_TYPE="${TASK_TYPE:-math}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher/${DATASET_NAME}/round1}"
python drpc/teacher/query_gpt.py \
  --dataset-name "$DATASET_NAME" --dataset-path "$DATASET_PATH" \
  --strategy "$STRATEGY" --model "$MODEL" --task-type "$TASK_TYPE" \
  --sample-index 0 --num-samples "$NUM_SAMPLES" \
  --output-dir "$OUTPUT_DIR" "$@"
