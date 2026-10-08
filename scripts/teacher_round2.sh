#!/bin/bash
# Stage 2: route every training question to the bank and let the teacher write the structured
# rationale along the retrieved path (the student training data). Requires OPENAI_API_KEY.
# Run from the repository root.
set -euo pipefail
QUESTIONS="${QUESTIONS:-data/GSM8K/train.jsonl}"
BANK="${BANK:-data/reasoning_bank/taxonomy.json}"
OUT_DIR="${OUT_DIR:-outputs/teacher/gsm8k/round2}"
MODEL="${MODEL:-gpt-5.1}"
TASK_TYPE="${TASK_TYPE:-math}"
NUM_QUESTIONS="${NUM_QUESTIONS:-10000}"
mkdir -p "$OUT_DIR"
python drpc/teacher/run_second_round_batch.py \
  --questions-path "$QUESTIONS" --reasoning-bank "$BANK" \
  --results-output "$OUT_DIR/second_round_results.json" \
  --new-routes-output "$OUT_DIR/new_routes.json" \
  --route-record "$OUT_DIR/route_record.json" \
  --num-questions "$NUM_QUESTIONS" --model "$MODEL" --task-type "$TASK_TYPE" "$@"
