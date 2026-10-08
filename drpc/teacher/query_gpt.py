#!/usr/bin/env python3
"""Query the OpenAI gpt5.1 model with GSM8K prompts and store responses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from drpc.teacher import prompts as prompt_module
from drpc.teacher.query_common import (
    DEFAULT_STRATEGY,
    apply_dataset_preset,
    build_common_arg_parser,
    build_prompt_bundle,
    extract_response_text,
    normalize_strategy,
    run_samples,
    sanitize_identifier,
    set_global_seed,
)

DEFAULT_MODEL = "gpt-4o-mini"


def ask_model(system_prompt: str, user_prompt: str, model_name: str, temperature: float | None) -> str:
    """Send the prompt text to the target model and return its response."""
    if OpenAI is None:
        raise SystemExit(
            "The `openai` package is required to query the model. "
            "Install it with `pip install openai`."
        )
    client = OpenAI()
    temp = 0.0 if temperature is None else float(temperature)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=42,
        temperature=temp,
    )
    return extract_response_text(response)

def build_arg_parser() -> argparse.ArgumentParser:
    return build_common_arg_parser(
        description="Combine GSM8K questions with utils.prompts and query gpt5.1.",
        default_model=DEFAULT_MODEL,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_dataset_preset(args)
    set_global_seed(42)
    strategy = normalize_strategy(args.strategy, default=DEFAULT_STRATEGY)
    if getattr(args, "category_based", None) is None:
        args.category_based = strategy == "rpb"
    system_prompt, user_prompt_builder = build_prompt_bundle(
        prompt_module,
        strategy,
        args.normal_shots,
        args.cot_shots,
        include_teacher_requirements=True,
        include_gold_answer=args.include_answer_in_prompt,
        task_type=args.task_type,
        category_based=args.category_based,
        detailed=args.detailed,
    )
    raw_output_path = None
    if strategy == "super_correct":
        safe_model = sanitize_identifier(args.model, "model", 0)
        raw_output_path = args.output_dir / f"{safe_model}_{strategy}"

    def _ask(system_prompt_text: str, user_prompt_text: str) -> str:
        return ask_model(system_prompt_text, user_prompt_text, args.model, args.temperature)

    run_samples(
        args=args,
        strategy=strategy,
        system_prompt=system_prompt,
        user_prompt_builder=user_prompt_builder,
        ask_fn=_ask,
        model_name=args.model,
        raw_output_path=raw_output_path,
    )


if __name__ == "__main__":
    main()
