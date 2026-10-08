#!/usr/bin/env python3
"""
Deterministic SG cleaning and validation (SGFT paper, arXiv:2412.09906v1, §3.1.3).

Rules enforced:
  - SG must be 2–6 steps.
  - SG must contain NO explicit arithmetic / calculation traces.
  - SG must NOT contain a resolved final-answer statement with a number.
  - SG must NOT contain a bare multiple-choice selection ("The answer is (C)").

CLI usage:
  python clean_sg.py --input data/sg_gsm8k.jsonl --output data/sg_gsm8k_clean.jsonl
"""

from __future__ import annotations

import argparse
import enum
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


_ARITH_RE = re.compile(r"\d+\s*[\+\-\*/]\s*\d+")

_EQ_NUM_RE = re.compile(r"\d[\d\s]*=\s*\d|=\s*\d[\d\s]*\d")

_FINAL_ANS_RE = re.compile(
    r"(therefore\s+the\s+answer\s+is"
    r"|the\s+answer\s+is"
    r"|final\s+answer[\s:=\-]"
    r"|answer[\s:=\-])"
    r"\s*[\$\£\€]?\s*\d",
    re.IGNORECASE,
)

_CHOICE_ANS_FINAL_RE = re.compile(
    r"(the\s+answer\s+is|answer\s+is)\s*[\(\[]?\s*[A-E]\s*[\)\]]?\s*[\.:]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_LONG_NUM_RE = re.compile(r"\b\d{5,}\b")

_UNICODE_MATH_RE = re.compile(r"[×÷∑∫√]|\d+\^[0-9]")

_STEP_LINE_RE = re.compile(
    r"^\s*"
    r"(?:step\s+\d+[\s:\-]*|(?:\d+[\.\):])\s+|[-•*]\s+)",
    re.IGNORECASE | re.MULTILINE,
)


class RejectReason(str, enum.Enum):
    EMPTY = "empty"
    TOO_FEW_STEPS = "too_few_steps"
    TOO_MANY_STEPS = "too_many_steps"
    CONTAINS_ARITHMETIC = "contains_arithmetic"
    CONTAINS_EQUATION = "contains_equation"
    CONTAINS_FINAL_ANSWER_NUMBER = "contains_final_answer_number"
    CONTAINS_CHOICE_ANSWER = "contains_choice_answer"
    CONTAINS_LONG_NUMBERS = "contains_long_numbers"
    CONTAINS_UNICODE_MATH = "contains_unicode_math"


def count_steps(sg: str) -> int:
    """
    Count steps in SG text.

    Tries structured step markers first (numbered / bulleted lines).
    Falls back to counting non-empty lines if fewer than 2 markers found.
    """
    if not sg or not sg.strip():
        return 0
    matches = _STEP_LINE_RE.findall(sg)
    if len(matches) >= 2:
        return len(matches)
    return sum(1 for ln in sg.splitlines() if ln.strip())


def validate_sg(sg: str) -> Tuple[bool, List[RejectReason]]:
    """
    Validate a single SG string.

    Returns:
        (True, []) if valid.
        (False, [reasons]) if invalid.
    """
    reasons: List[RejectReason] = []

    if not sg or not sg.strip():
        return False, [RejectReason.EMPTY]

    n = count_steps(sg)
    if n < 2:
        reasons.append(RejectReason.TOO_FEW_STEPS)
    if n > 6:
        reasons.append(RejectReason.TOO_MANY_STEPS)

    if _ARITH_RE.search(sg):
        reasons.append(RejectReason.CONTAINS_ARITHMETIC)
    if _EQ_NUM_RE.search(sg):
        reasons.append(RejectReason.CONTAINS_EQUATION)
    if _FINAL_ANS_RE.search(sg):
        reasons.append(RejectReason.CONTAINS_FINAL_ANSWER_NUMBER)
    if _CHOICE_ANS_FINAL_RE.search(sg):
        reasons.append(RejectReason.CONTAINS_CHOICE_ANSWER)
    if _LONG_NUM_RE.search(sg):
        reasons.append(RejectReason.CONTAINS_LONG_NUMBERS)
    if _UNICODE_MATH_RE.search(sg):
        reasons.append(RejectReason.CONTAINS_UNICODE_MATH)

    return len(reasons) == 0, reasons


def clean_file(
    input_path: Path,
    output_path: Path,
    rejected_path: Optional[Path] = None,
    sg_field: str = "sg",
) -> Tuple[int, int, int]:
    """
    Read JSONL, validate each row's SG field, write valid rows to output_path
    and rejected rows (with reason codes) to rejected_path.

    Returns (total, accepted, rejected) counts.
    """
    total = accepted = rejected = 0
    out_fh = output_path.open("w", encoding="utf-8")
    rej_fh = (rejected_path or Path(os.devnull)).open("w", encoding="utf-8") \
             if rejected_path else None

    import os
    try:
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                sg = obj.get(sg_field, "")
                valid, reasons = validate_sg(sg)
                if valid:
                    accepted += 1
                    out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    rejected += 1
                    obj["_reject_reasons"] = [r.value for r in reasons]
                    if rej_fh:
                        rej_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        out_fh.close()
        if rej_fh:
            rej_fh.close()

    return total, accepted, rejected


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deterministic SG cleaning/validation for SGFT data."
    )
    p.add_argument("--input", type=Path, required=True, help="Input JSONL with SG data.")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL with valid SG rows.")
    p.add_argument(
        "--rejected",
        type=Path,
        default=None,
        help="Optional JSONL path for rejected rows with reason codes.",
    )
    p.add_argument(
        "--sg-field",
        default="sg",
        help="Name of the SG field in each JSON row (default: 'sg').",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total, accepted, rejected = clean_file(
        args.input,
        args.output,
        rejected_path=args.rejected,
        sg_field=args.sg_field,
    )
    pct = accepted / total * 100 if total else 0.0
    print(
        f"[clean_sg] total={total}  accepted={accepted} ({pct:.1f}%)  rejected={rejected}",
        flush=True,
    )


if __name__ == "__main__":
    main()
