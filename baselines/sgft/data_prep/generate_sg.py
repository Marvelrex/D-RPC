#!/usr/bin/env python3
"""
SG Data Generation using a GPT teacher model (zero-shot).

Uses prompt_ii (zero-shot) from Table 1 of arXiv:2412.09906v1 for every
training example — no seed/few-shot examples are shown to the teacher.

Input:    local JSONL file via --train-file
Teacher:  GPT-5.2 (configurable via --teacher)
N:        all rows in --train-file by default (configurable via --n)

Output JSONL schema (per row):
  {
    "id":                "<row index>",
    "question":          "...",
    "response_rationale":"Step 1: ..."
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from clean_sg import validate_sg, count_steps


_GUIDELINE_RULE = (
    " Write in plain English only."
    " Do not perform any calculations, do not write any equations, formulas, or mathematical symbols."
    " Each step should describe what to do conceptually, not how to compute it."
)

PROMPT_II_AQUA = (
    "Output 2 to 6 high-level solution guidelines for this multiple-choice problem."
    " The steps should guide toward identifying and selecting the correct option."
    + _GUIDELINE_RULE
)

PROMPT_II_GSM8K = (
    "Output 2 to 6 high-level solution guidelines for this problem."
    " The final answer is a decimal number."
    + _GUIDELINE_RULE
)

PROMPT_II_STRATEGYQA = (
    "Output 2 to 6 high-level solution guidelines for this problem."
    " The final answer is either True or False."
    + _GUIDELINE_RULE
)

PROMPT_II_DEFAULT = (
    "Output 2 to 6 high-level solution guidelines for this problem."
    + _GUIDELINE_RULE
)

DATASET_PROMPTS = {
    "aqua":        PROMPT_II_AQUA,
    "gsm8k":       PROMPT_II_GSM8K,
    "strategyqa":  PROMPT_II_STRATEGYQA,
}

RETRY_CONSTRAINT = (
    "Do not include any calculations, equations, formulas, or mathematical symbols."
    " Use plain English only to describe what to do conceptually at a high level."
)


def _append_options(question: str, row: dict) -> str:
    """Append options/choices to question text if present (AQUA-style rows)."""
    for key in ("options", "Options", "option", "Option"):
        val = row.get(key)
        if not val:
            continue
        opts_text = str(val).strip()
        if not opts_text or opts_text in question:
            continue
        joiner = "" if question.endswith("\n") else "\n"
        return f"{question}{joiner}{opts_text}"
    return question


def load_local_jsonl(path: Path, dataset_label: str) -> List[dict]:
    """Load a local JSONL file; each output row has index, dataset, question."""
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = None
            for key in ("index", "id", "qid", "question_id", "questionId"):
                if key in obj and obj[key] is not None:
                    idx = str(obj[key])
                    break
            if idx is None:
                idx = f"{dataset_label}_{i:06d}"
            question = str(obj.get("question", "")).strip()
            if not question:
                continue
            question = _append_options(question, obj)
            rows.append({"index": idx, "dataset": dataset_label,
                         "split": "train", "question": question})
    return rows


def _init_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai SDK: pip install openai>=1.0")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "  export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=key)


def call_gpt(client, model: str, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [API] attempt {attempt+1}/{max_retries} failed: {exc}. "
                      f"Retrying in {wait}s …", flush=True)
                time.sleep(wait)
            else:
                raise
    return ""


def generate_training_sg(
    client,
    teacher: str,
    questions: List[dict],
    n: int,
    output_path: Path,
    dataset: str,
) -> None:
    """Generate SG for N rows using prompt_ii (zero-shot). Resumes if interrupted."""
    selected = questions[:n]

    existing: Dict[str, dict] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing[obj.get("id", obj.get("index", ""))] = obj
                except Exception:
                    continue
        if existing:
            print(f"  [RESUME] {len(existing)} examples already done.", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_fh = output_path.open("a", encoding="utf-8")

    try:
        for i, q_row in enumerate(selected):
            idx = q_row["index"]
            if idx in existing:
                continue

            question = q_row["question"]
            prompt_ii = DATASET_PROMPTS.get(dataset.lower(), PROMPT_II_DEFAULT)
            prompt = f"{prompt_ii}\n\nProblem: {question}"
            print(f"\n{'='*60}", flush=True)
            print(f"  [{i+1}/{n}] {dataset} | id={idx}", flush=True)
            print(f"  [PROMPT]\n{prompt}", flush=True)
            sg_text = call_gpt(client, teacher, prompt)
            print(f"  [RESPONSE]\n{sg_text}", flush=True)

            valid, reasons = validate_sg(sg_text)

            if not valid:
                retry_prompt = prompt + "\n" + RETRY_CONSTRAINT
                print(f"  [RETRY] {[r.value for r in reasons]}", flush=True)
                print(f"  [RETRY PROMPT]\n{retry_prompt}", flush=True)
                sg_text = call_gpt(client, teacher, retry_prompt)
                print(f"  [RETRY RESPONSE]\n{sg_text}", flush=True)
                valid, _ = validate_sg(sg_text)

            print(f"  [SAVED] id={idx}", flush=True)
            out_fh.write(json.dumps({
                "id": idx,
                "question": question,
                "response_rationale": sg_text,
            }, ensure_ascii=False) + "\n")
            out_fh.flush()
    finally:
        out_fh.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate zero-shot SG training data from a local JSONL file."
    )
    p.add_argument("--dataset", required=True,
                   help="Label for the dataset (used in output filenames, e.g. gsm8k).")
    p.add_argument("--train-file", type=Path, required=True,
                   help="Path to local training JSONL. Each row needs a 'question' field.")
    p.add_argument("--n", type=int, default=None,
                   help="Number of entries to process. "
                        "If larger than the dataset size, all rows are processed. "
                        "Already-generated entries are skipped automatically. "
                        "Default: process all rows.")
    p.add_argument("--teacher", default="gpt-5.2",
                   help="OpenAI model name (default: gpt-5.2).")
    p.add_argument("--output-dir", type=Path, default=Path("Baselines/SGFT/data"),
                   help="Output directory (default: Baselines/SGFT/data).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (currently unused; kept for reproducibility logging).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _init_client()

    print(f"\n[1/2] Loading {args.dataset} from {args.train_file} …", flush=True)
    train_rows = load_local_jsonl(args.train_file, args.dataset)
    print(f"      Loaded {len(train_rows)} rows.", flush=True)

    n = min(args.n, len(train_rows)) if args.n is not None else len(train_rows)
    print(f"[SGFT] dataset={args.dataset}  n={n}/{len(train_rows)}  teacher={args.teacher}", flush=True)

    sg_path = out_dir / f"sg_{args.dataset}.jsonl"
    print(f"\n[2/2] Generating {n} SGs → {sg_path}", flush=True)

    generate_training_sg(
        client=client,
        teacher=args.teacher,
        questions=train_rows,
        n=n,
        output_path=sg_path,
        dataset=args.dataset,
    )

    count = sum(1 for ln in sg_path.open("r", encoding="utf-8") if ln.strip())
    print(f"\n[SGFT] Done. {count} SGs saved to {sg_path}", flush=True)


if __name__ == "__main__":
    main()
