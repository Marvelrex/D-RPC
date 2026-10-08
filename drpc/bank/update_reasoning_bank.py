#!/usr/bin/env python3
"""
Incrementally add a new (category, intent, reasoning_path) route into the reasoning bank.

Usage:
python helper/update_reasoning_bank.py --route-file new_route.json \
    --reasoning-bank data/reasoning_bank/taxonomy.json

Where new_route.json can be either:
- A full output object with a top-level "route" key, or
- A dict with keys: category (str), intent (str or list[str]), reasoning_path (list[str]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_route(route_obj: dict) -> tuple[str, str, List[str]]:
    route = route_obj.get("route", route_obj)
    category = str(route.get("category") or "").strip()
    intents = route.get("intent") or []
    if isinstance(intents, list) and intents:
        intent = str(intents[0]).strip()
    else:
        intent = str(intents).strip()
    reasoning_path = route.get("reasoning_path") or []
    if not (category and intent and isinstance(reasoning_path, list) and reasoning_path):
        raise ValueError("Route must contain category, intent, and non-empty reasoning_path list.")
    return category, intent, reasoning_path


def compute_summary(taxonomy: Dict[str, Dict[str, List[List[str]]]]) -> Dict[str, int]:
    categories = set(taxonomy.keys())
    intents = set()
    reasoning_paths = set()
    for cat, intent_map in taxonomy.items():
        for intent, paths in intent_map.items():
            intents.add((cat, intent))
            for p in paths:
                reasoning_paths.add((cat, intent, tuple(p)))
    return {
        "unique_categories": len(categories),
        "unique_intents": len(intents),
        "unique_reasoning_paths": len(reasoning_paths),
    }


def update_reasoning_bank(reasoning_bank_path: Path, category: str, intent: str, reasoning_path: List[str]) -> None:
    data = load_json(reasoning_bank_path)
    has_wrapper = False
    if "taxonomy" in data and isinstance(data["taxonomy"], dict):
        taxonomy = data["taxonomy"]
        has_wrapper = True
    else:
        taxonomy = data

    taxonomy.setdefault(category, {})
    taxonomy[category].setdefault(intent, [])
    paths = taxonomy[category][intent]
    if tuple(reasoning_path) not in [tuple(p) for p in paths]:
        paths.append(reasoning_path)

    summary = compute_summary(taxonomy)

    if has_wrapper:
        data["taxonomy"] = taxonomy
        data["summary"] = summary
        out_obj = data
    else:
        out_obj = taxonomy

    reasoning_bank_path.parent.mkdir(parents=True, exist_ok=True)
    with reasoning_bank_path.open("w", encoding="utf-8") as fh:
        json.dump(out_obj, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a new route to the reasoning bank.")
    parser.add_argument("--route-file", type=Path, required=True, help="Path to JSON with route or {route: {...}}.")
    parser.add_argument(
        "--reasoning-bank",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Path to taxonomy/reasoning bank JSON.",
    )
    args = parser.parse_args()

    route_obj = load_json(args.route_file)
    category, intent, reasoning_path = normalize_route(route_obj)
    update_reasoning_bank(args.reasoning_bank, category, intent, reasoning_path)
    print(f"Added route to category='{category}', intent='{intent}' with reasoning_path={reasoning_path} in {args.reasoning_bank}")


if __name__ == "__main__":
    main()
