#!/bin/bash
#SBATCH --job-name=drpc_ablation
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
# usage: [MODEL=...] sbatch --export=ALL run_ablation.slurm <DS> <RUN> <DATA> <TRAIN_SIZE> <TESTFILE> <MAXGEN> <OUTROOT> [SEED]
# Same recipe as the paper runs (LoRA r=64 alpha=128, lr 1e-4, batch 2x8, 2 epochs, greedy, 2048 new tokens).
# Re-submitting the same cell resumes from the saved checkpoint and the rows already generated.
set -euo pipefail
source "${VENV_ACTIVATE:?set VENV_ACTIVATE to your virtualenv activate script}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"; mkdir -p logs

DS="$1"; RUN="$2"; DATA="$3"; TS="$4"; TESTFILE="$5"; MAXGEN="$6"; OUTROOT="$7"; SEED="${8:-42}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MSLUG="${MODEL##*/}"
OUT="${OUTROOT}/${DS}/${MSLUG}/rpb/Run${RUN}"
mkdir -p "${OUTROOT}/sigs" "${OUT}/pred"
SIG="${OUTROOT}/sigs/signature_${DS}_${MSLUG}_rpb_Run${RUN}.json"

TARGET=$(wc -l < "$TESTFILE"); [ "$MAXGEN" -lt "$TARGET" ] && TARGET=$MAXGEN
PRED=$(find "${OUT}/pred" -name "rpb_*.jsonl" | head -1 || true)
if [ -n "$PRED" ] && [ "$(wc -l < "$PRED")" -ge "$TARGET" ]; then echo "already complete"; exit 0; fi

python drpc/student/run_distill_experiments.py \
  --strategy rpb --data-file "$DATA" --train-sizes "$TS" \
  --model-name "$MODEL" --tokenizer-name "$MODEL" \
  --uniform-signature-file "$SIG" --output-root "$OUT" --pred-root "${OUT}/pred" \
  --save-steps 250 --logging-steps 25 --warmup-ratio 0.03 \
  --learning-rate 1e-4 --batch-size 2 --grad-accum 8 --num-epochs 2 \
  --optim adamw_torch --lr-scheduler-type linear --max-grad-norm 1.0 \
  --lora-r 64 --lora-alpha 128 --seed "$SEED" \
  --generate --no-do-sample --temperature 0.0 \
  --max-gen-samples "$MAXGEN" --max-new-tokens 2048 \
  --test-file "$TESTFILE"

PRED=$(find "${OUT}/pred" -name "rpb_*.jsonl" | head -1)
python drpc/eval/eval_json_accuracy.py "$PRED" --format-validity | tail -1
