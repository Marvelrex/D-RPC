#!/usr/bin/env python3
"""Build prompts for every (dataset, strategy) we plan to run pass@k on
and assert they're well-formed (e.g., MCQ datasets must include options).
No GPU / no model load — uses the same prompt builders as eval_passk.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from drpc.student.distill_rationale import (
    _choice_schema_for_path,
    _normalize_task_type,
    build_instructions_with_source,
    build_messages_for_inference,
    load_jsonl,
)
from drpc.eval.eval_passk import build_dcot_prompt

DATASETS = {
    "AQUA":       (REPO_ROOT / "data/AQUA/gsm8k_format_test.jsonl",      "mcq",  re.compile(r"\bA\)")),
    "AI2ARC":     (REPO_ROOT / "data/AI2ARC/gsm8k_format_test.jsonl",    "mcq",  re.compile(r"\bA\)")),
    "GPQA":       (REPO_ROOT / "data/GPQA/gsm8k_format_test.jsonl",      "mcq",  re.compile(r"\bA\)")),
    "GSM8K":      (REPO_ROOT / "data/GSM8K/test.jsonl",                  "num",  None),
    "MATH":       (REPO_ROOT / "data/MATH/gsm8k_format_test_5000.jsonl", "num",  None),
    "StrategyQA": (REPO_ROOT / "data/StrategyQA/dev_gsm8k_format.jsonl", "bool", None),
}

STRATEGIES = ("rpb", "cot", "freeform", "super_correct", "dcot", "sgft")


def build_prompt_for(strategy: str, row: dict, test_file: Path) -> str:
    if strategy == "dcot":
        return build_dcot_prompt(row, num_cots=3)
    task_type = _normalize_task_type(None, test_file)
    choice_schema = _choice_schema_for_path(test_file)
    instructions, _ = build_instructions_with_source(
        strategy, detailed=True, task_type=task_type, choice_schema=choice_schema
    )
    msgs = build_messages_for_inference(row, instructions)
    return f"[SYSTEM]\n{msgs[0]['content']}\n\n[USER]\n{msgs[1]['content']}"


def validate(ds: str, strategy: str) -> tuple[bool, str]:
    test_file, kind, mcq_re = DATASETS[ds]
    rows = load_jsonl(test_file, limit=1)
    if not rows:
        return False, "no rows in test file"
    row = rows[0]
    try:
        prompt = build_prompt_for(strategy, row, test_file)
    except Exception as exc:
        return False, f"build failed: {type(exc).__name__}: {exc}"

    q = (row.get("question") or "").strip()
    if q[:40] not in prompt:
        return False, "question text missing"

    if kind == "mcq":
        opts = row.get("options")
        if mcq_re and not mcq_re.search(prompt):
            return False, f"MCQ options A)/B)/... not found (options={opts!r})"

    if strategy == "rpb" and ds in ("AQUA", "AI2ARC", "GPQA"):
        if '"ans"' not in prompt:
            return False, "RPB JSON spec missing"
    if strategy == "cot":
        if "step" not in prompt.lower() and "let" not in prompt.lower():
            return False, "CoT prompt missing 'let's think step by step' wording"
    if strategy == "dcot":
        if "[Number of answers]" not in prompt:
            return False, "DCoT prompt missing [Number of answers]"

    return True, "OK"


def main() -> None:
    print(f"{'dataset':12s}  {'strategy':14s}  status")
    print("-" * 70)
    fails: list[tuple[str, str, str]] = []
    for ds in DATASETS:
        for s in STRATEGIES:
            ok, msg = validate(ds, s)
            tag = "PASS" if ok else "FAIL"
            print(f"{ds:12s}  {s:14s}  {tag:5s} {msg}")
            if not ok:
                fails.append((ds, s, msg))
    print("-" * 70)
    print(f"{len(fails)} failures")
    if "--dump" in sys.argv:
        outdir = REPO_ROOT / "outputs" / "prompt_samples"
        outdir.mkdir(parents=True, exist_ok=True)
        for ds in DATASETS:
            for s in STRATEGIES:
                try:
                    rows = load_jsonl(DATASETS[ds][0], limit=1)
                    p = build_prompt_for(s, rows[0], DATASETS[ds][0])
                    (outdir / f"{ds}_{s}.txt").write_text(p, encoding="utf-8")
                except Exception as exc:
                    (outdir / f"{ds}_{s}.ERROR.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        print(f"[DUMP] wrote prompts to {outdir}")


if __name__ == "__main__":
    main()
