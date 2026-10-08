#!/bin/bash
# Decompress the released teacher data in place (keeps the .gz files). Run from the repository root.
set -euo pipefail
for f in data/*/teacher/*.gz supplementary/bank_ablation/data/*/*.gz baselines/sgft/data/*.gz; do
  [ -e "${f%.gz}" ] || gunzip -k "$f"
done
echo "done"
