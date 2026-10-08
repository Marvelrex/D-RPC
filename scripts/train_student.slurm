#!/bin/bash
#SBATCH --job-name=drpc_student
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/student_%j.out
#SBATCH --error=logs/student_%j.err
# Stage 3 + 4: LoRA-distil one student on one teacher file, then generate greedy predictions on the
# test set. One submission = one run; submit with RUN=1..10 for the 10-run protocol (seed 42 fixed).
# Example: STRATEGY=rpb DATA_FILE=data/MATH/teacher/rpb_second_round.json TRAIN_SIZE=10000 \
#          TEST_FILE=data/MATH/gsm8k_format_test_5000.jsonl MODEL=Qwen/Qwen3-1.7B RUN=1 sbatch scripts/train_student.slurm
set -euo pipefail
source "${VENV_ACTIVATE:?set VENV_ACTIVATE to your virtualenv activate script}"
cd "${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; mkdir -p logs

STRATEGY="${STRATEGY:-rpb}"
DATA_FILE="${DATA_FILE:?teacher data file, e.g. data/GSM8K/teacher/rpb_second_round.json}"
TRAIN_SIZE="${TRAIN_SIZE:-10000}"
TEST_FILE="${TEST_FILE:?test file, e.g. data/GSM8K/test.jsonl}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RUN="${RUN:-1}"
SEED="${SEED:-42}"
MAX_GEN="${MAX_GEN:-$(wc -l < "$TEST_FILE")}"
DETAILED="${DETAILED:-}"
TAG="$(basename "$(dirname "$(dirname "$DATA_FILE")")")_${STRATEGY}"
OUT="${OUTPUT_ROOT:-outputs/students}/${TAG}/${MODEL##*/}/Run${RUN}"
mkdir -p "$OUT/pred"

python drpc/student/run_distill_experiments.py \
  --strategy "$STRATEGY" --data-file "$DATA_FILE" --train-sizes "$TRAIN_SIZE" \
  --model-name "$MODEL" --tokenizer-name "$MODEL" \
  --uniform-signature-file "$OUT/signature.json" --output-root "$OUT" --pred-root "$OUT/pred" \
  --save-steps 250 --logging-steps 25 --warmup-ratio 0.03 \
  --learning-rate 1e-4 --batch-size 2 --grad-accum 8 --num-epochs 2 \
  --optim adamw_torch --lr-scheduler-type linear --max-grad-norm 1.0 \
  --lora-r 64 --lora-alpha 128 --seed "$SEED" ${DETAILED:+--detailed} \
  --generate --no-do-sample --temperature 0.0 \
  --max-gen-samples "$MAX_GEN" --max-new-tokens 2048 \
  --test-file "$TEST_FILE"

PRED=$(find "$OUT/pred" -name "${STRATEGY}_*.jsonl" | head -1)
python drpc/eval/eval_json_accuracy.py "$PRED" --format-validity | tail -1
