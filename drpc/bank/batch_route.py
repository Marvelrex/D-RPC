#!/usr/bin/env python3
"""
Batch route GSM8K questions to (category, intent, reasoning_path) using a reasoning bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from drpc.bank.router import route_question, load_reasoning_bank


def load_questions(path: Path) -> List[str]:
    questions: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question")
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch route GSM8K questions to reasoning paths.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("GSM8K/test.jsonl"),
        help="Path to GSM8K test JSONL file.",
    )
    parser.add_argument(
        "--reasoning-bank",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Path to reasoning bank JSON (or taxonomy.json with top-level 'taxonomy').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reasoning_bank/routed_test.json"),
        help="Where to save routed results.",
    )
    args = parser.parse_args()

    reasoning_bank = load_reasoning_bank(args.reasoning_bank)
    if "taxonomy" in reasoning_bank:
        reasoning_bank = reasoning_bank["taxonomy"]

    questions = load_questions(args.questions_path)
    results = []
    for q in questions:
        routed = route_question(q, reasoning_bank)
        results.append({"question": q, **routed})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} routed questions to {args.output}")


if __name__ == "__main__":
    main()
