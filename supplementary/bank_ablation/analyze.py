#!/usr/bin/env python3
"""Score a bank-ablation campaign against the full D-RPC baseline and write a markdown report.

Per cell (dataset, arm, model): accuracy of each run on the question ids shared by all arm runs
and all baseline runs; gap = mean(arm) - mean(baseline) with a Welch 95% confidence interval over
runs. Pooled: per model, the three schema-control arms (30 run accuracies) vs the baseline runs;
"both" averages the two model gaps. Predictions are read as predictions.jsonl(.gz).
"""
import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from drpc.eval.eval_json_accuracy import evaluate_with_labels  # noqa: E402

MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-1.7B"]
CONTROLS = ["empty", "random", "randglobal"]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179,
        14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 100: 1.984}


def t975(df: float) -> float:
    keys = sorted(T975)
    if df >= keys[-1]:
        return T975[keys[-1]]
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return T975[lo]
    return T975[lo] + (T975[hi] - T975[lo]) * (df - lo) / (hi - lo)


def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se2 > 0 else 1.0
    h = t975(df) * math.sqrt(se2)
    return ma - mb, ma - mb - h, ma - mb + h


def labels(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    _, _, _, lab = evaluate_with_labels(rows)
    return {r["id"]: l[2] for r, l in zip(rows, lab)}


def runs(dir_: Path):
    files = sorted(dir_.glob("Run*/predictions.jsonl*"))
    return [labels(f) for f in files]


def accs(run_labels, ids):
    return [100.0 * sum(r[i] for i in ids) / len(ids) for r in run_labels]


def fmt(g, lo, hi):
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{g:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="subdirectory of predictions/, e.g. seed42 or seeds43-52")
    parser.add_argument("--datasets", nargs="+", default=["MATH", "AQUA"])
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "predictions/table1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    pred_root = Path(__file__).resolve().parent / "predictions" / args.campaign
    out = args.output or Path(__file__).resolve().parent / "results" / f"{args.campaign}.md"

    lines = [f"# Bank-ablation campaign `{args.campaign}` vs full D-RPC baseline", "",
             "gap = arm - baseline on the question ids shared by all runs; Welch 95% CI over runs; * = CI excludes 0.", ""]
    per_run = []
    for ds in args.datasets:
        arms = sorted(p.name for p in (pred_root / ds).iterdir() if p.is_dir())
        lines += [f"## {ds}", "", "| model | arm | n ids | baseline mean±std | arm mean±std | gap [95% CI] |", "|---|---|---|---|---|---|"]
        pooled = {}
        for m in MODELS:
            base = runs(args.baseline_root / ds / "RPB" / m / "lora")
            shared_base = set.intersection(*[set(r) for r in base])
            arm_runs = {arm: runs(pred_root / ds / arm / m) for arm in arms}
            per_run.append((ds, m, "baseline", accs(base, shared_base), len(shared_base)))
            for arm in arms:
                ids = shared_base.intersection(*[set(r) for r in arm_runs[arm]])
                ra, ba = accs(arm_runs[arm], ids), accs(base, ids)
                g, lo, hi = welch(ra, ba)
                per_run.append((ds, m, arm, ra, len(ids)))
                lines.append(f"| {m} | {arm} | {len(ids)} | {statistics.mean(ba):.2f}±{statistics.stdev(ba):.2f} | "
                             f"{statistics.mean(ra):.2f}±{statistics.stdev(ra):.2f} | {fmt(g, lo, hi)} |")
            ctrl = [a for a in CONTROLS if a in arm_runs]
            if ctrl:
                ids = shared_base.intersection(*[set(r) for a in ctrl for r in arm_runs[a]])
                pooled[m] = welch([x for a in ctrl for x in accs(arm_runs[a], ids)], accs(base, ids))
        if len(pooled) == 2:
            (gL, loL, hiL), (gQ, loQ, hiQ) = pooled[MODELS[0]], pooled[MODELS[1]]
            hL, hQ = (hiL - loL) / 2, (hiQ - loQ) / 2
            g, h = (gL + gQ) / 2, math.sqrt(hL ** 2 + hQ ** 2) / 2
            lines += ["", f"Pooled schema controls ({'+'.join(ctrl)}): {MODELS[0]} {fmt(gL, loL, hiL)}; "
                          f"{MODELS[1]} {fmt(gQ, loQ, hiQ)}; both {fmt(g, g - h, g + h)}", ""]
    lines += ["## Per-run accuracy", ""]
    for ds in args.datasets:
        lines += [f"### {ds}", "", "| model | arm | n ids | " + " | ".join(f"R{i}" for i in range(1, 11)) + " | mean | std |",
                  "|---|---|---|" + "---|" * 12]
        for d, m, arm, a, n in per_run:
            if d == ds:
                lines.append(f"| {m} | {arm} | {n} | " + " | ".join(f"{x:.2f}" for x in a) +
                             f" | {statistics.mean(a):.2f} | {statistics.stdev(a):.2f} |")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
