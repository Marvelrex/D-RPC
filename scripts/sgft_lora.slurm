#!/bin/bash
#SBATCH --job-name=drpc_sgft
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sgft_%j.out
#SBATCH --error=logs/sgft_%j.err
# SGFT baseline, stage 1: LoRA-train the student on the SGFT teacher data (baselines/sgft/data/sg_<dataset>.jsonl)
# with the shared recipe; stage 2 (collaborative inference) is baselines/sgft/sgft_stage2.py --help.
# Example: DATASET=gsm8k TEST_FILE=data/GSM8K/test.jsonl MODEL=Qwen/Qwen3-1.7B RUN=1 sbatch scripts/sgft_lora.slurm
set -euo pipefail
source "${VENV_ACTIVATE:?set VENV_ACTIVATE to your virtualenv activate script}"
cd "${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; mkdir -p logs

DATASET="${DATASET:-gsm8k}"
DATA_FILE="${DATA_FILE:-baselines/sgft/data/sg_${DATASET}.jsonl}"
TRAIN_SIZE="${TRAIN_SIZE:-$(python -c "import sys;print(sum(1 for _ in open(sys.argv[1])))" "$DATA_FILE")}"
TEST_FILE="${TEST_FILE:?test file}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RUN="${RUN:-1}"
OUT="${OUTPUT_ROOT:-outputs/students}/${DATASET}_sgft/${MODEL##*/}/Run${RUN}"
mkdir -p "$OUT/pred"

STRATEGY=sgft DATA_FILE="$DATA_FILE" TRAIN_SIZE="$TRAIN_SIZE" TEST_FILE="$TEST_FILE" MODEL="$MODEL" RUN="$RUN" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/students}" bash scripts/train_student.slurm
