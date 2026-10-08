#!/usr/bin/env python3
"""
Sample a fixed number of rows from a dataset proportionally to category counts.

Supports JSON arrays or JSONL inputs. The category field can be nested via dot
notation (e.g., "category_intent.category"). Output is JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from drpc.bank.sampling_utils import proportional_allocation


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    records: List[Dict[str, Any]] = []
    if not text:
        return records
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            records = [r for r in obj if isinstance(r, dict)]
            return records
        if isinstance(obj, dict):
            records = [obj]
            return records
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


def get_nested(d: Dict[str, Any], key_path: str) -> Any:
    cur: Any = d
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def proportional_sample(records: List[Dict[str, Any]], cat_key: str, target: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cat_val = get_nested(r, cat_key)
        cat = "" if cat_val is None else str(cat_val)
        by_cat.setdefault(cat, []).append(r)
    cats = list(by_cat.keys())
    counts = [len(by_cat[c]) for c in cats]
    if sum(counts) == 0:
        return []
    alloc = proportional_allocation(counts, target)

    sampled: List[Dict[str, Any]] = []
    for cat, n in zip(cats, alloc):
        pool = by_cat[cat]
        random.shuffle(pool)
        sampled.extend(pool[: min(n, len(pool))])
    return sampled


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample records proportionally by category.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL path.")
    parser.add_argument("--output", type=Path, help="Output JSONL path (default: <input>_sampled_<N>.jsonl).")
    parser.add_argument("--target-size", type=int, default=100, help="Number of records to sample (default: 100).")
    parser.add_argument(
        "--category-key",
        type=str,
        default="category",
        help='Category field (dot-separated for nested dicts; default: "category").',
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise SystemExit(f"No records loaded from {args.input}")

    sampled = proportional_sample(records, args.category_key, args.target_size, args.seed)

    out_path = args.output or args.input.with_name(f"{args.input.stem}_sampled_{args.target_size}.jsonl")
    save_jsonl(out_path, sampled)

    cnt = Counter([str(get_nested(r, args.category_key) or "") for r in sampled])
    print(f"Saved {len(sampled)} records to {out_path}")
    for cat, v in cnt.most_common():
        pct = v / len(sampled) * 100 if sampled else 0
        print(f"{cat or '<blank>'}: {v} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
