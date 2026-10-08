#!/bin/bash
# Submit one job per incomplete ablation cell (4 arms x 2 datasets x 2 models x 10 runs) and keep
# re-submitting until every prediction file is complete. Skips cells whose job is already queued
# or whose prediction file changed in the last 20 minutes. Set SEED_MODE=fixed for the paper
# protocol (seed 42 for every run) or SEED_MODE=per-run for seed = 42 + run.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUTROOT="${OUTROOT:?set OUTROOT to the output directory}"
SLURM="$PROJECT_ROOT/supplementary/bank_ablation/run_ablation.slurm"
SEED_MODE="${SEED_MODE:-fixed}"
LOG="$OUTROOT/keeper.log"
cd "$PROJECT_ROOT"
declare -A MODELS=([Llama]="meta-llama/Llama-3.1-8B-Instruct" [Qwen]="Qwen/Qwen3-1.7B")
ARMS=(
 "MATH_empty|supplementary/bank_ablation/data/MATH/empty.json|10000|data/MATH/gsm8k_format_test_5000.jsonl|5000"
 "MATH_random|supplementary/bank_ablation/data/MATH/random.json|10000|data/MATH/gsm8k_format_test_5000.jsonl|5000"
 "MATH_randglobal|supplementary/bank_ablation/data/MATH/randglobal.json|10000|data/MATH/gsm8k_format_test_5000.jsonl|5000"
 "MATH_filtered|supplementary/bank_ablation/data/MATH/filtered.json|7482|data/MATH/gsm8k_format_test_5000.jsonl|5000"
 "AQUA_empty|supplementary/bank_ablation/data/AQUA/empty.json|10000|data/AQUA/gsm8k_format_test.jsonl|254"
 "AQUA_random|supplementary/bank_ablation/data/AQUA/random.json|10000|data/AQUA/gsm8k_format_test.jsonl|254"
 "AQUA_randglobal|supplementary/bank_ablation/data/AQUA/randglobal.json|10000|data/AQUA/gsm8k_format_test.jsonl|254"
 "AQUA_filtered|supplementary/bank_ablation/data/AQUA/filtered.json|8827|data/AQUA/gsm8k_format_test.jsonl|254"
)
mkdir -p "$OUTROOT"
echo "$(date '+%F %T') keeper start (pid $$)" >> "$LOG"
while true; do
  queued=$(squeue -u "$USER" -h -o "%j")
  done=0; submitted=0
  for arm in "${ARMS[@]}"; do IFS='|' read ds data ts test maxgen <<< "$arm"
    for mk in Llama Qwen; do
      for run in $(seq 1 10); do
        [ "$SEED_MODE" = fixed ] && seed=42 || seed=$((42 + run))
        out="$OUTROOT/$ds/${MODELS[$mk]##*/}/rpb/Run$run"
        pred=$(find "$out/pred" -name "rpb_*.jsonl" 2>/dev/null | head -1)
        rows=0; [ -n "$pred" ] && rows=$(wc -l < "$pred")
        if [ "$rows" -ge "$maxgen" ]; then done=$((done + 1)); continue; fi
        name="abl_${ds}_${mk}_R${run}"
        grep -qxF "$name" <<< "$queued" && continue
        [ -n "$pred" ] && [ -n "$(find "$pred" -mmin -20 2>/dev/null)" ] && continue
        sbatch --job-name="$name" --export=ALL,MODEL="${MODELS[$mk]}" "$SLURM" "$ds" "$run" "$data" "$ts" "$test" "$maxgen" "$OUTROOT" "$seed" > /dev/null
        submitted=$((submitted + 1))
      done
    done
  done
  echo "$(date '+%F %T') complete=$done/160 submitted=$submitted" >> "$LOG"
  [ "$done" -ge 160 ] && { echo "$(date '+%F %T') all cells complete" >> "$LOG"; break; }
  sleep 300
done
