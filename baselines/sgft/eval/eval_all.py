#!/usr/bin/env python3
"""
Evaluation for SGFT collaborative inference predictions.

Dataset-specific evaluators:
  GSM8K      — exact match on extracted final numeric answer
  AQUA-RAT   — exact match on extracted multiple-choice letter (A–E)
  StrategyQA — exact match on extracted yes/no boolean

Usage:
  # Evaluate a single predictions file:
  python eval_all.py --predictions path/to/predictions.jsonl --dataset gsm8k

  # Evaluate all JSONL files in a directory (auto-detect dataset from filename):
  python eval_all.py --predictions-dir Baselines/SGFT/outputs/ \
                     --output-csv Baselines/SGFT/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _norm_number(s: str) -> str:
    """Strip commas, trailing zeros after decimal, whitespace."""
    s = str(s).strip().replace(",", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        return s.lower()


def _norm_choice(s: str) -> str:
    return str(s).strip().upper()[:1]


def _norm_bool(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("yes", "true", "1"):
        return "yes"
    if s in ("no", "false", "0"):
        return "no"
    return s


NORMALIZERS = {
    "gsm8k": _norm_number,
    "aqua": _norm_choice,
    "strategyqa": _norm_bool,
}


def evaluate_file(
    predictions_path: Path,
    dataset: Optional[str] = None,
) -> Dict[str, object]:
    """
    Evaluate a JSONL predictions file.

    Returns a dict with:
      accuracy, correct, total, dataset, combo_id, guide_ckpt, path
    """
    rows: List[dict] = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return {
            "accuracy": 0.0, "correct": 0, "total": 0,
            "dataset": dataset or "unknown",
            "combo_id": "unknown", "guide_ckpt": "unknown",
            "path": str(predictions_path),
        }

    if dataset is None:
        dataset = rows[0].get("dataset", "")
    if not dataset:
        name = predictions_path.stem.lower()
        for ds in ("gsm8k", "aqua", "strategyqa"):
            if ds in name:
                dataset = ds
                break
        dataset = dataset or "gsm8k"

    norm = NORMALIZERS.get(dataset, lambda x: str(x).strip().lower())
    combo_id = rows[0].get("combo_id", "unknown")
    guide_ckpt = rows[0].get("guide_ckpt", "unknown")

    correct = total = 0
    for row in rows:
        gold = row.get("gold_ans", "")
        pred = row.get("model_ans", "")
        if pred is None:
            pred = ""
        total += 1
        if norm(str(pred)) == norm(str(gold)):
            correct += 1

    accuracy = correct / total if total else 0.0
    return {
        "accuracy": round(accuracy * 100, 2),
        "correct": correct,
        "total": total,
        "dataset": dataset,
        "combo_id": combo_id,
        "guide_ckpt": guide_ckpt,
        "path": str(predictions_path),
    }


def print_markdown_table(results: List[Dict]) -> None:
    if not results:
        print("No results to display.")
        return
    headers = ["dataset", "combo_id", "accuracy (%)", "correct", "total", "guide_ckpt"]
    rows = [
        [
            str(r.get("dataset", "")),
            str(r.get("combo_id", ""))[:40],
            f"{r.get('accuracy', 0.0):.2f}",
            str(r.get("correct", 0)),
            str(r.get("total", 0)),
            str(r.get("guide_ckpt", ""))[:30],
        ]
        for r in results
    ]
    col_widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    print(header_row)
    print(sep)
    for row in rows:
        print("| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")


def save_csv(results: List[Dict], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "combo_id", "accuracy", "correct", "total", "guide_ckpt", "path"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[eval] CSV saved to {path}", flush=True)


def save_json(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] JSON saved to {path}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SGFT collaborative inference predictions.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--predictions",
        type=Path,
        help="Path to a single predictions JSONL file.",
    )
    grp.add_argument(
        "--predictions-dir",
        type=Path,
        help="Directory containing *.jsonl prediction files (evaluated recursively).",
    )
    p.add_argument(
        "--dataset",
        choices=["gsm8k", "aqua", "strategyqa"],
        default=None,
        help="Override dataset label (auto-detected from file content if omitted).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Save summary table as CSV.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save summary table as JSON.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.predictions:
        files = [args.predictions]
    else:
        files = sorted(args.predictions_dir.rglob("*.jsonl"))
        if not files:
            print(f"No .jsonl files found in {args.predictions_dir}")
            return

    results = []
    for f in files:
        print(f"[eval] Evaluating {f} …", flush=True)
        r = evaluate_file(f, dataset=args.dataset)
        results.append(r)
        print(
            f"  dataset={r['dataset']}  combo={r['combo_id']}  "
            f"acc={r['accuracy']:.2f}%  ({r['correct']}/{r['total']})",
            flush=True,
        )

    print("\n" + "=" * 70)
    print("SGFT Evaluation Summary")
    print("=" * 70)
    print_markdown_table(results)

    if args.output_csv:
        save_csv(results, args.output_csv)
    if args.output_json:
        save_json(results, args.output_json)


if __name__ == "__main__":
    main()
