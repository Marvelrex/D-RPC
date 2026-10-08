#!/bin/bash
# Stage 1: compress the round-1 rpb rationales into the Category -> Intent -> reasoning-path bank.
# Run from the repository root.
set -euo pipefail
RESULTS="${RESULTS:-outputs/teacher/gsm8k/round1/results_rpb.jsonl}"
BANK="${BANK:-data/reasoning_bank/taxonomy.json}"
mkdir -p "$(dirname "$BANK")"
python drpc/bank/build_taxonomy.py --results-path "$RESULTS" --output-path "$BANK" "$@"
python drpc/bank/filter_taxonomy_similar.py --taxonomy "$BANK" --output "$BANK"
