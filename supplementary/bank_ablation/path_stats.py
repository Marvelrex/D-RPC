#!/usr/bin/env python3
"""Reasoning-path reuse and entropy of the teacher training data, per dataset and arm.

path = tuple of step names. bank = path given to the teacher (reasoning_path_bank.reasoning_path),
used = keys of the rationale the teacher wrote, step = individual step names. Items with an empty
path are excluded. reuse = 1 - distinct/n; share2 = fraction of items whose path occurs at least
twice; H = Shannon entropy in bits; H/log2k = normalised entropy; perplexity = 2^H.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ["empty", "random", "randglobal", "filtered"]


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def entropy(counter):
    n = sum(counter.values())
    return -sum(v / n * math.log2(v / n) for v in counter.values()) if n else 0.0


def stats(paths):
    c = collections.Counter(paths)
    n, k = len(paths), len(c)
    h = entropy(c)
    return dict(n=n, k=k, reuse=1 - k / n, share2=sum(v for v in c.values() if v >= 2) / n, H=h,
                Hn=h / math.log2(k) if k > 1 else 0.0, ppl=2 ** h, top1=c.most_common(1)[0][1] / n)


def row(ds, arm, which, s):
    return (f"| {ds} | {arm} | {which} | {s['n']} | {s['k']} | {s['reuse']:.3f} | {s['share2']:.3f} | "
            f"{s['H']:.2f} | {s['Hn']:.3f} | {s['ppl']:.1f} | {s['top1']:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["MATH", "AQUA"])
    parser.add_argument("--output", type=Path, default=HERE / "results" / "path_stats.md")
    args = parser.parse_args()
    lines = ["# Reasoning-path reuse and entropy (teacher training data)", "",
             "path = tuple of step names; bank = path given to the teacher; used = keys of the written rationale; "
             "step = individual step names. reuse = 1 - distinct/n; share>=2 = fraction of items whose path occurs "
             "at least twice; H = Shannon entropy (bits); H/log2k = normalised; perplexity = 2^H; top1 = share of the most frequent path.", "",
             "| dataset | arm | which | n | distinct | reuse | share>=2 | H (bits) | H/log2k | perplexity | top1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in args.datasets:
        sources = {"baseline": ROOT / "data" / ds / "teacher" / "rpb_second_round.json.gz"}
        for arm in ARMS:
            sources[arm] = HERE / "data" / ds / f"{arm}.json.gz"
        for arm, path in sources.items():
            if not path.exists():
                continue
            items = load(path)
            bank = [tuple(it["reasoning_path_bank"].get("reasoning_path") or []) for it in items]
            used = [tuple((it.get("rationale") or {}).keys()) for it in items]
            for which, paths in [("bank path", bank), ("used path", used),
                                 ("bank step", [s for p in bank for s in p]), ("used step", [s for p in used for s in p])]:
                paths = [p for p in paths if p]
                if paths:
                    lines.append(row(ds, arm, which, stats(paths)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
