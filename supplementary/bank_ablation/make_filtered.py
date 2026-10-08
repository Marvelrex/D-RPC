#!/usr/bin/env python3
"""Build the FILTERED arm: the full D-RPC teacher data without the items whose teacher answer
does not match the gold answer (ans_matches_gold == False). MATH 10000 -> 7482, AQUA 10000 -> 8827."""
import argparse
import gzip
import json
from pathlib import Path


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = load(args.source)
    kept = [it for it in items if it.get("ans_matches_gold") is True]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=1)
    print(f"{len(kept)}/{len(items)} items kept -> {args.output}")


if __name__ == "__main__":
    main()
