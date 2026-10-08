#!/usr/bin/env python3
"""
Query GPT-5.1 to produce category/intent JSON for each question in a dataset using CATEGORY_INTENT_PROMPT.
Outputs <dataset_name>_category.jsonl alongside the dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from drpc.teacher import prompts as prompt_module
from drpc.teacher.query_common import load_samples, extract_response_text

DEFAULT_MODEL = "gpt-5.1"


def ask_model(prompt: str, model_name: str, temperature: float | None) -> str:
    client = OpenAI()
    temp = 0.0 if temperature is None else float(temperature)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        seed=42,
        temperature=temp,
    )
    return extract_response_text(resp)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate category/intent annotations with GPT-5.1.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the input JSONL dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier to query (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to write the annotated JSONL (default: <dataset_name>_category.jsonl next to input).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based row index to start from (default: 0).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of consecutive samples to process (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = args.dataset_path
    if args.output_path:
        out_path = args.output_path
    else:
        out_path = dataset.with_name(f"{dataset.stem}_category.jsonl")

    samples = load_samples(dataset, args.sample_index, args.num_samples)

    prompt_template = (prompt_module.CATEGORY_INTENT_PROMPT or "").strip()
    if not prompt_template:
        raise SystemExit("CATEGORY_INTENT_PROMPT is empty in prompts.py.")

    processed_ids = set()
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as existing:
                for line in existing:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = obj.get("qid") or obj.get("id") or obj.get("index")
                    if qid is not None:
                        processed_ids.add(str(qid))
            if processed_ids:
                print(f"Found {len(processed_ids)} processed records in {out_path}; will skip them.")
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    mode = "a" if processed_ids else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for offset, (abs_idx, sample) in enumerate(samples, start=1):
            question = sample.get("question", "").strip()
            sample_id = sample.get("id") or sample.get("index") or f"{dataset.stem}_{abs_idx:05d}"
            if str(sample_id) in processed_ids:
                print(f"Skipping already-processed sample {offset}/{args.num_samples} (id={sample_id})")
                continue
            prompt = f"{prompt_template}\n{question}"
            print(f"Processing sample {offset}/{args.num_samples} (id={sample_id})")
            print("---- Prompt ----")
            print(prompt)
            print("---- Prompt End ----")
            response = ask_model(prompt, args.model, args.temperature)
            try:
                parsed = json.loads(response)
            except Exception:
                parsed = {"raw_response": response}
            record = {
                "qid": sample.get("id") or sample.get("index") or sample.get("qid"),
                "question": question,
                "answer": sample.get("answer"),
                "category_intent": parsed,
            }
            print("Category/intent result:")
            print(json.dumps(record, ensure_ascii=False, indent=2))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            added += 1
    print(f"Wrote {added} new record(s) to {out_path}")


if __name__ == "__main__":
    main()
