#!/usr/bin/env python3
"""Rebuild the Table-1 accuracy summary from the released prediction files.

Scores every predictions/table1/<dataset>/<method>/<model>/lora/Run*/predictions.jsonl(.gz)
with eval_json_accuracy.evaluate and prints mean +- sample std over runs as a markdown table.
"""
import argparse
import gzip
import json
import statistics
from pathlib import Path

from drpc.eval.eval_json_accuracy import evaluate

DATASETS = ["GSM8K", "AQUA", "StrategyQA", "AI2ARC", "MATH"]
METHODS = ["CoT", "Freeform", "SuperCorrect", "DCoT", "SGFT", "RPB"]
MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-1.7B"]


def read_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def accuracy(path: Path) -> float:
    correct, total, _ = evaluate(read_rows(path))
    return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("predictions/table1"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    for model in args.models:
        print(f"\n### {model}\n")
        print("| dataset | " + " | ".join(METHODS) + " |")
        print("|---|" + "---|" * len(METHODS))
        for dataset in DATASETS:
            cells = []
            for method in METHODS:
                files = sorted((args.root / dataset / method / model / "lora").glob("Run*/predictions.jsonl*"))
                if not files:
                    cells.append("-")
                    continue
                accs = [accuracy(f) for f in files]
                std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                cells.append(f"{statistics.mean(accs):.2f} ± {std:.2f} (n={len(accs)})")
            print(f"| {dataset} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
