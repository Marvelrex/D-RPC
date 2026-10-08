#!/usr/bin/env python3
"""Query a local Hugging Face Llama model with GSM8K prompts and store responses."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch as th
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drpc.teacher import prompts as prompt_module
from drpc.teacher.query_common import (
    DEFAULT_STRATEGY,
    apply_dataset_preset,
    build_common_arg_parser,
    build_prompt_bundle,
    normalize_strategy,
    run_samples,
    set_global_seed,
)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def ask_model(
        system_prompt: str,
        user_prompt: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: th.device,
        temperature: float | None,
        do_sample: bool | None,
) -> str:
    """Send the prompt text to the target model and return its response."""
    system_msg = {"role": "system", "content": system_prompt.strip()}
    user_content = user_prompt.strip() + "\nReturn ONLY valid JSON with keys rationale and ans."
    user_msg = {"role": "user", "content": user_content}
    messages = [system_msg, user_msg]

    if getattr(tokenizer, "chat_template", None):
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_string = f"<s>[INST] <<SYS>>\n{system_prompt.strip()}\n<</SYS>>\n\n{user_prompt.strip()} [/INST]"

    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)

    eos_id = tokenizer.eos_token_id
    eot_id = None
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        pass

    eos_token_ids = []
    if isinstance(eos_id, list):
        eos_token_ids.extend(eos_id)
    elif eos_id is not None:
        eos_token_ids.append(eos_id)
    if eot_id is not None and eot_id not in eos_token_ids and eot_id != -1:
        eos_token_ids.append(eot_id)
    eos_arg = eos_token_ids if eos_token_ids else eos_id

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    temp = 0.0 if temperature is None else float(temperature)
    sample = False if do_sample is None else bool(do_sample)
    generation_kwargs = dict(
        max_new_tokens=512,
        do_sample=sample,
        eos_token_id=eos_arg,
        pad_token_id=pad_id,
        min_new_tokens=1,
    )
    if sample:
        generation_kwargs["temperature"] = max(temp, 1e-5)
        generation_kwargs["top_p"] = 1.0

    outputs = model.generate(**inputs, **generation_kwargs)

    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return response


def build_arg_parser() -> argparse.ArgumentParser:
    return build_common_arg_parser(
        description="Combine GSM8K questions with prompts and query a local Llama model.",
        default_model=DEFAULT_MODEL,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_dataset_preset(args)

    strategy = normalize_strategy(args.strategy, default=DEFAULT_STRATEGY)
    if getattr(args, "category_based", None) is None:
        args.category_based = strategy == "rpb"
    system_prompt, user_prompt_builder = build_prompt_bundle(
        prompt_module,
        strategy,
        args.normal_shots,
        args.cot_shots,
        include_teacher_requirements=False,
        include_gold_answer=args.include_answer_in_prompt,
        task_type=args.task_type,
        category_based=args.category_based,
        detailed=args.detailed,
    )

    show_only = args.show_only

    if not show_only:
        seed = int(os.environ.get("LLM_QUERY_SEED", "18"))
        set_global_seed(seed)
        print(f"Using fixed seed {seed}")
        print(f"Loading model: {args.model}")
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            dtype=th.bfloat16,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        temp = 0.0 if args.temperature is None else float(args.temperature)
        do_sample = False if args.do_sample is None else bool(args.do_sample)
        model.generation_config.do_sample = do_sample
        if do_sample:
            model.generation_config.temperature = max(temp, 1e-5)

        model.to(device)
        model.eval()
        print(f"Model loaded on device: {device}")

        def _ask(system_prompt_text: str, user_prompt_text: str) -> str:
            with th.no_grad():
                return ask_model(
                    system_prompt_text,
                    user_prompt_text,
                    model,
                    tokenizer,
                    device,
                    args.temperature,
                    args.do_sample,
                )
    else:
        device = th.device("cpu")
        tokenizer = None

        def _ask(system_prompt_text: str, user_prompt_text: str) -> str:
            raise RuntimeError("Model invocation skipped because --show-only is set.")

    run_samples(
        args=args,
        strategy=strategy,
        system_prompt=system_prompt,
        user_prompt_builder=user_prompt_builder,
        ask_fn=_ask,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
