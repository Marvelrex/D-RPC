#!/usr/bin/env python3
"""
Distill the filtered rationale datasets into a compact Llama 3.2 1B model using a chosen strategy
(normal, super_correct, RPB, or freeform).

Training uses full-parameter SFT + AdamW (no LoRA adapters) so only the
supervision/prompt strategy differs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys
import re

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drpc.teacher import prompts as prompt_module

from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_LR_SCHEDULER = "linear"
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 42

SGFT_PROMPT_IV = (
    "Please generate a step-by-step solution for the following problem with no calculations.\n"
    " You don't need to solve it, just output the steps in 2 to 6 steps."
)

NORMAL_DATA_FILE = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_normal.jsonl"
SUPER_CORRECT_DATA_FILE = REPO_ROOT / "data" / "structure_rationale" / "results_super_correct_filtered.jsonl"
STRUCT_BASE_DIR = REPO_ROOT / "filtered_gpt5_rationales"
INTERSECTION_IDS = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_intersection_ids.txt"

STRATEGY_TO_DIR: Dict[str, str] = {
    "rpb": "freeform",
}

TEXT_TASK_TYPES = {"text", "commonsense", "common", "strategyqa", "strategy_qa"}
CHOICE_DATASET_SCHEMAS = {
    "aqua": "<A|B|C|D|E>",
}
DATASET_TASK_TYPE_MAP = {
    "strategyqa": "text",
    "aqua": "math",
    "gsm8k": "math",
    "svamp": "math",
}


def _normalize_task_type(task_type: Optional[str], data_path: Optional[Path]) -> str:
    """Normalize task type, auto-detecting StrategyQA-style paths as text."""
    if task_type:
        return task_type.strip().lower()
    if data_path:
        path_text = str(data_path).casefold()
        for name, mapped in DATASET_TASK_TYPE_MAP.items():
            if name.casefold() in path_text:
                return mapped
    return "math"


def _choice_schema_for_path(data_path: Optional[Path]) -> Optional[str]:
    if not data_path:
        return None
    path_text = str(data_path).casefold()
    for name, schema in CHOICE_DATASET_SCHEMAS.items():
        if name in path_text:
            return schema
    return None


def _apply_choice_schema(prompt_text: str, choices_schema: Optional[str]) -> str:
    if not prompt_text or not choices_schema:
        return prompt_text
    choices = [c.strip() for c in choices_schema.strip("<>").split("|") if c.strip()]
    if len(choices) > 1:
        choices_str = ", ".join(choices[:-1]) + f", or {choices[-1]}"
    elif choices:
        choices_str = choices[0]
    else:
        choices_str = choices_schema
    replacements = {
        "<numeric>": choices_schema,
        "<numeric final answer>": choices_schema,
        "<bool>": choices_schema,
        '"ans": <numeric>': f'"ans": {choices_schema}',
        '"ans": <bool>': f'"ans": {choices_schema}',
        '"ans" must be a numeric value (integer or decimal), not a string.': f'"ans" must be one of {choices_str}.',
        '- "ans" must be a numeric value (integer or decimal), not a string.': f'- "ans" must be one of {choices_str}.',
        '"ans" must be numeric (no strings, units, or words).': f'"ans" must be one of {choices_str}.',
        '- "ans" must be numeric (no strings, units, or words).': f'- "ans" must be one of {choices_str}.',
        '"ans" must be numeric.': f'"ans" must be one of {choices_str}.',
        '- "ans" must be numeric.': f'- "ans" must be one of {choices_str}.',
        '"ans" must be true or false.': f'"ans" must be one of {choices_str}.',
        '- "ans" must be true or false.': f'- "ans" must be one of {choices_str}.',
    }
    updated = prompt_text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def _adapt_prompt_for_task_type(prompt_text: str, task_type: str, choice_schema: Optional[str] = None) -> str:
    """Swap numeric answer instructions to boolean or choice schema for text/commonsense or MC tasks."""
    if not prompt_text:
        return prompt_text
    updated = prompt_text
    if task_type.lower() in TEXT_TASK_TYPES:
        replacements = {
            "<numeric>": "<bool>",
            "<numeric final answer>": "<bool>",
            '"ans": <numeric>': '"ans": <bool>',
            '"ans" must be a numeric value (integer or decimal), not a string.': '"ans" must be true or false.',
            '- "ans" must be a numeric value (integer or decimal), not a string.': '- "ans" must be true or false.',
            '"ans" must be numeric (no strings, units, or words).': '"ans" must be true or false.',
            '- "ans" must be numeric (no strings, units, or words).': '- "ans" must be true or false.',
            '"ans" must be numeric.': '"ans" must be true or false.',
            '- "ans" must be numeric.': '- "ans" must be true or false.',
            '"ans" must be numeric': '"ans" must be true or false',
        }
        for old, new in replacements.items():
            updated = updated.replace(old, new)
    updated = _apply_choice_schema(updated, choice_schema)
    return updated


def _resolve_prompt_from_module(names: list[str]) -> str:
    """Return the first present, non-empty prompt text for any of the given names."""
    for name in names:
        if hasattr(prompt_module, name):
            val = getattr(prompt_module, name)
            if isinstance(val, str) and val.strip():
                return val.strip()
    raise AttributeError(f"None of the prompt names were found: {names}")


STRATEGY_TO_PROMPT_NAMES: Dict[str, list[str]] = {
    "rpb": [
        "STRUCTURED_FREE_FORM_PART_THREE",
        "Structured_RPB_PART_THREE",
    ],
    "freeform": [
        "FREEFORM_REASONING_PATH_PART_THREE",
        "FREEFORM_REASONING_PATH_PART_THREE_DETAILED",
    ],
}
DETAILED_STRATEGY_TO_PROMPT_NAMES: Dict[str, list[str]] = {
    "rpb": [
        "Structured_RPB_PART_THREE_DETAILED",
        "STRUCTURED_FREEFORM_PART_THREE_DETAILED",
        "STRUCTURED_FREE_FORM_PART_THREE_DETAILED",
        "Structured_RPB_PART_THREE",
        "STRUCTURED_FREE_FORM_PART_THREE",
    ],
    "freeform": [
        "FREEFORM_REASONING_PATH_PART_THREE_DETAILED",
        "FREEFORM_REASONING_PATH_PART_THREE",
    ],
}

QUESTION_PLACEHOLDER = "<QUESTION_TEXT>"
NO_SYSTEM_PREFIX = "__NO_SYSTEM__\n"
PLAIN_TARGET_FLAG = "__PLAIN_TARGET__\n"


@dataclass(frozen=True)
class UniformTrainingConfig:
    """Training hyperparameters that must be identical for every strategy run."""
    model_name: str
    tokenizer_name: str
    max_length: int
    learning_rate: float
    weight_decay: float
    num_epochs: float
    batch_size: int
    grad_accum: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    optim: str
    bf16: bool
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    lr_scheduler_type: str
    max_grad_norm: float
    seed: int
    pad_to_max: bool


@dataclass
class DistillConfig:
    data_file: Path
    output_dir: Path
    strategy: str
    task_type: str
    detailed: bool
    intersection_file: Optional[Path]
    max_samples: Optional[int]
    train_size: Optional[int]
    uniform: UniformTrainingConfig
    cot_shots: int
    generate: bool
    test_file: Path
    gen_output_file: Path
    max_gen_samples: Optional[int]
    max_new_tokens: int
    temperature: float
    do_sample: bool
    flatten_targets: bool
    print_chat: bool
    signature_file: Path
    reset_signature: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill rationale datasets with full-parameter SFT (no LoRA adapters).")
    parser.add_argument(
        "--strategy",
        type=str.lower,
        choices=["normal", "rpb", "super_correct", "freeform", "cot", "sgft"],
        default="rpb",
        help="Prompt strategy to distill (normal, super_correct, RPB, or freeform).",
    )
    parser.add_argument(
        "--task-type",
        type=str.lower,
        default=None,
        help="Task type for prompt formatting (math vs text/commonsense). Defaults to math unless data_file hints at StrategyQA.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use detailed rationale prompts when available (structured/freeform strategies).",
    )
    parser.add_argument("--data-file", type=Path, help="Path to MathQA rationale JSONL (overrides strategy default).")
    parser.add_argument("--output-dir", type=Path, help="Directory to store the finetuned model (overrides strategy default).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HF model id to finetune (default: Llama 3.2 1B Instruct).")
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name/id to reuse for every strategy (default: Llama 3.1 8B Instruct tokenizer).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of training samples (shared across strategies).")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Number of examples to actually use for training after filtering/formatting (required for parity).")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length for training examples.")
    parser.add_argument(
        "--cot-shots",
        type=int,
        default=2,
        help="Number of few-shot examples to include for strategy=cot (0-8, default 2).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--num-epochs", type=float, default=2.0, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Fraction of steps used for LR warmup.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging interval (in steps).")
    parser.add_argument("--save-steps", type=int, default=250, help="Checkpoint interval (in steps).")
    parser.add_argument(
        "--optim",
        type=str,
        choices=["adamw_torch"],
        default="adamw_torch",
        help="Optimizer to use (restricted to AdamW for uniformity).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        choices=["linear"],
        default=DEFAULT_LR_SCHEDULER,
        help="LR scheduler type (fixed to linear for cross-strategy parity).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM, help="Gradient clipping norm.")
    parser.add_argument("--bf16", action="store_true", help="Train using bfloat16 (otherwise fp16).")
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="Compatibility flag from LoRA script; ignored in full-SFT mode.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="Compatibility flag from LoRA script; ignored in full-SFT mode.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Compatibility flag from LoRA script; ignored in full-SFT mode.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed to reuse for every strategy.")
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad all sequences to max_length (default: pad to longest in batch).",
    )
    parser.add_argument(
        "--intersection-file",
        type=Path,
        default=None,
        help="Optional list of IDs to keep (RPB only; disabled by default).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After training, run generation on a test file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=REPO_ROOT / "data" / "test.jsonl",
        help="Test JSONL path for generation.",
    )
    parser.add_argument(
        "--gen-output-file",
        type=Path,
        default=None,
        help="Where to write generations (default: <output_dir>/predictions.jsonl).",
    )
    parser.add_argument(
        "--max-gen-samples",
        type=int,
        default=300,
        help="Limit number of test rows to generate (default: 300).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature when --generate is enabled (default: 0.0).",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling when generating (--generate).",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling when generating (--generate).",
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--flatten-targets",
        action="store_true",
        help="Flatten rationale/ans fields into plain text before building the target JSON (default: off).",
    )
    parser.add_argument(
        "--print-chat",
        action="store_true",
        help="Print the formatted chat string used for tokenization for every example.",
    )
    parser.add_argument(
        "--uniform-signature-file",
        type=Path,
        default=SCRIPTS_DIR / "uniform_training_signature.json",
        help="Where to record and validate shared training hyperparameters across strategies.",
    )
    parser.add_argument(
        "--reset-uniform-signature",
        action="store_true",
        help="Overwrite the recorded shared-hyperparameter signature (use only when intentionally changing it).",
    )
    return parser


def slugify(text: str) -> str:
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def log_trainable_parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Log the number of trainable vs total parameters for reproducibility checks."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable_params / total_params * 100) if total_params else 0.0
    print(
        f"[PARAMS] trainable={trainable_params:,} total={total_params:,} ({pct:.6f}% trainable)",
        flush=True,
    )
    return trainable_params, total_params


def _extract_final_number(text: str) -> str:
    if not isinstance(text, str):
        return ""
    token = text.strip().split()[-1].strip(",. ")
    return token


def _uniform_signature(cfg: DistillConfig) -> Dict[str, object]:
    """Generate a signature capturing every training hyperparameter that must stay uniform.

    Note: question IDs are injected by callers to tie strategies to the exact same data slice.
    """
    uni = cfg.uniform
    return {
        "model_name": uni.model_name,
        "tokenizer_name": uni.tokenizer_name,
        "max_length": uni.max_length,
        "learning_rate": uni.learning_rate,
        "weight_decay": uni.weight_decay,
        "num_epochs": uni.num_epochs,
        "batch_size": uni.batch_size,
        "grad_accum": uni.grad_accum,
        "warmup_ratio": uni.warmup_ratio,
        "logging_steps": uni.logging_steps,
        "save_steps": uni.save_steps,
        "optim": uni.optim,
        "bf16": uni.bf16,
        "use_lora": uni.use_lora,
        "lora_r": uni.lora_r,
        "lora_alpha": uni.lora_alpha,
        "lora_dropout": uni.lora_dropout,
        "lora_target_modules": list(uni.lora_target_modules),
        "lr_scheduler_type": uni.lr_scheduler_type,
        "max_grad_norm": uni.max_grad_norm,
        "seed": uni.seed,
        "pad_to_max": uni.pad_to_max,
        "max_samples": cfg.max_samples,
    }


def _write_signature_atomic(sig_path: Path, signature: Dict[str, object]) -> None:
    """Write signature JSON atomically to avoid partial/empty files on interruptions."""
    payload = json.dumps(signature, indent=2)
    tmp_path = sig_path.with_suffix(sig_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(sig_path)


def _load_signature_dict(sig_path: Path) -> Dict[str, object] | None:
    """Load an existing signature file; return None when missing/empty/corrupt."""
    try:
        raw = sig_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[UNIFORM] Warning: failed reading signature file {sig_path}: {exc}", flush=True)
        return None
    if not raw:
        print(f"[UNIFORM] Warning: signature file is empty: {sig_path}", flush=True)
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[UNIFORM] Warning: signature JSON is invalid at {sig_path}: {exc}", flush=True)
        return None
    if not isinstance(parsed, dict):
        print(f"[UNIFORM] Warning: signature must be a JSON object, got {type(parsed).__name__}", flush=True)
        return None
    return parsed


def enforce_uniform_signature(cfg: DistillConfig, train_question_ids: List[str]) -> None:
    """Persist and verify a shared hyperparameter signature across strategy runs."""
    sig_path = cfg.signature_file
    sig = _uniform_signature(cfg)
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    if sig_path.exists() and not cfg.reset_signature:
        recorded = _load_signature_dict(sig_path)
        if recorded is None:
            _write_signature_atomic(sig_path, sig)
            print(f"[UNIFORM] Recreated shared training signature at {sig_path}", flush=True)
            return
        recorded.pop("data_file", None)
        recorded_filtered = {k: v for k, v in recorded.items() if k in sig}
        if recorded_filtered != sig:
            diffs = [k for k in sig if recorded_filtered.get(k) != sig[k]]
            raise ValueError(
                "Per-strategy hyperparameter override detected. "
                f"Mismatched keys: {', '.join(diffs)}. "
                f"To intentionally change the shared setup, rerun with --reset-uniform-signature "
                f"after deleting {sig_path}."
            )
        _write_signature_atomic(sig_path, sig)
        print(f"[UNIFORM] Verified shared training signature at {sig_path}", flush=True)
    else:
        _write_signature_atomic(sig_path, sig)
        print(f"[UNIFORM] Recorded shared training signature at {sig_path}", flush=True)


def assert_uniform_training(uniform_cfg: UniformTrainingConfig) -> None:
    """Ensure shared optimizer/training settings are strategy-invariant."""
    optim_lower = uniform_cfg.optim.lower()
    if "adamw" not in optim_lower:
        raise ValueError(f"Optimizer must be AdamW for every strategy (got {uniform_cfg.optim}).")
    if uniform_cfg.use_lora:
        if uniform_cfg.lora_target_modules != DEFAULT_LORA_TARGET_MODULES:
            raise ValueError("LoRA target modules must stay identical across strategies.")
        if uniform_cfg.lora_r <= 0 or uniform_cfg.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and shared across strategies.")
        if not (0.0 <= uniform_cfg.lora_dropout < 1.0):
            raise ValueError("LoRA dropout must be in [0,1) for all strategies.")
    if uniform_cfg.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and shared across strategies.")


def log_uniform_config(cfg: DistillConfig, prompt_source: str | None = None) -> None:
    """Print the shared training configuration so parity is obvious in logs."""
    print("[UNIFORM] Using shared training hyperparameters:", flush=True)
    signature = _uniform_signature(cfg)
    for key, val in sorted(signature.items()):
        print(f"  - {key}: {val}", flush=True)
    print(f"[UNIFORM] Task type: {getattr(cfg, 'task_type', 'math')}", flush=True)
    if prompt_source:
        print(f"[UNIFORM] Prompt source: {prompt_source}", flush=True)


def extract_question_id(row: dict) -> str:
    """Return a stable question identifier; raises if none is present."""
    for key in ("index", "id", "question_id", "questionId"):
        if key in row:
            return str(row[key])
    raise ValueError("Each row must include an 'index' or 'id' field for cross-strategy alignment.")


def prepare_training_rows(rows: List[dict], train_size: int) -> tuple[List[dict], List[str]]:
    """Order rows by question id and select the shared training subset."""
    rows_with_ids: list[tuple[str, dict]] = []
    for row in rows:
        qid = extract_question_id(row)
        rows_with_ids.append((qid, row))
    all_ids = [qid for qid, _ in rows_with_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate question IDs detected; ensure dataset has unique IDs before training.")
    rows_with_ids.sort(key=lambda pair: pair[0])
    if train_size > len(rows_with_ids):
        raise ValueError(f"train_size={train_size} exceeds available unique questions ({len(rows_with_ids)}).")
    selected = rows_with_ids[:train_size]
    question_ids = [qid for qid, _ in selected]
    selected_rows = [row for _, row in selected]
    assert len(selected_rows) == train_size, "Selected rows mismatch requested train_size."
    return selected_rows, question_ids


def _stringify_row(row: dict) -> dict:
    """Convert all values to strings (JSON-dumping dict/list) to keep Arrow columns homogenous."""
    out: dict = {}
    for key, val in row.items():
        if val is None:
            out[key] = ""
        elif isinstance(val, (dict, list)):
            out[key] = json.dumps(val, ensure_ascii=False)
        elif isinstance(val, str):
            out[key] = val
        else:
            out[key] = str(val)
    return out


def _coerce_plain_text(value: object, key_order: Optional[List[str]] = None) -> str:
    """Flatten structured rationale/answer fields into plain sentences.

    key_order can be provided to preserve the original JSON key order when a dict
    has been reordered by downstream tooling (e.g., HF Dataset/Arrow).
    """
    def _with_period(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text.endswith((".", "!", "?")):
            return text
        return text + "."

    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
            return _coerce_plain_text(parsed, key_order=key_order)
        except Exception:
            return stripped
    if isinstance(value, dict):
        parts: List[str] = []
        ordered_keys = key_order if key_order else list(value.keys())
        seen: set[str] = set()
        for k in ordered_keys:
            if k not in value:
                continue
            seen.add(k)
            text = _coerce_plain_text(value[k])
            if text:
                parts.append(_with_period(f"{k}: {text}"))
        for k, v in value.items():
            if k in seen:
                continue
            text = _coerce_plain_text(v)
            if text:
                parts.append(_with_period(f"{k}: {text}"))
        return " ".join(p for p in parts if p).strip()
    if isinstance(value, list):
        parts = [_with_period(_coerce_plain_text(v)) for v in value]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()


def _reorder_dict(d: Dict[str, object], key_order: Optional[List[str]]) -> Dict[str, object]:
    """Return a shallow copy of d ordered by key_order (extras appended)."""
    if not key_order:
        return {k: v for k, v in d.items() if v is not None}
    ordered: Dict[str, object] = {k: d[k] for k in key_order if k in d}
    for k, v in d.items():
        if k not in ordered:
            if v is None:
                continue
            ordered[k] = v
    return ordered


def _clean_answer(value: object) -> object:
    """Prefer numeric answers; fall back to plain text if parsing fails."""
    if isinstance(value, (int, float)):
        return value
    try:
        as_float = float(str(value).strip())
        if as_float.is_integer():
            return int(as_float)
        return as_float
    except Exception:
        return _coerce_plain_text(value)


def _extract_options_text(example: dict) -> str:
    """Return options/choices text if present in common fields."""
    for key in ("options", "Options", "option", "Option"):
        val = example.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def _append_options(question_text: str, options_text: str) -> str:
    """Append options to question text if not already included."""
    if not options_text:
        return question_text.rstrip()
    base = question_text.rstrip()
    if options_text in base:
        return base
    joiner = "" if base.endswith("\n") else "\n"
    return f"{base}{joiner}{options_text}"


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            iterable = parsed
        elif isinstance(parsed, dict):
            iterable = [parsed]
        else:
            iterable = []
        for obj in iterable:
            if not isinstance(obj, dict):
                continue
            if limit is not None and len(rows) >= limit:
                break
            if "response_rationale" not in obj and "rationale" in obj:
                obj["response_rationale"] = obj["rationale"]
            if "response_ans" not in obj and "ans" in obj:
                obj["response_ans"] = obj["ans"]
            rat = obj.get("response_rationale")
            if isinstance(rat, dict):
                obj["_rationale_key_order"] = list(rat.keys())
            rows.append(obj)
        if rows:
            return rows
    except Exception:
        pass

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if "response_rationale" not in obj and "rationale" in obj:
                    obj["response_rationale"] = obj["rationale"]
                if "response_ans" not in obj and "ans" in obj:
                    obj["response_ans"] = obj["ans"]
                rat = obj.get("response_rationale")
                if isinstance(rat, dict):
                    obj["_rationale_key_order"] = list(rat.keys())
                rows.append(obj)
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_instructions_with_source(
    strategy: str,
    cot_shots: int | None = None,
    detailed: bool = False,
    task_type: str = "math",
    choice_schema: Optional[str] = None,
) -> tuple[str, str]:
    strategy = strategy.lower().strip()
    is_text_task = task_type.lower() in TEXT_TASK_TYPES
    if strategy == "normal":
        if hasattr(prompt_module, "NORMAL_PART_THREE"):
            part_three = str(getattr(prompt_module, "NORMAL_PART_THREE", "")).strip()
        elif hasattr(prompt_module, "normalize_part_three"):
            part_three = str(prompt_module.normalize_part_three()).strip()
        else:
            raise ValueError("Normal strategy requires NORMAL_PART_THREE or normalize_part_three in prompts.py")
        part_two = _adapt_prompt_for_task_type(prompt_module.PART_TWO_TASK.strip(), task_type, choice_schema)
        part_three = _adapt_prompt_for_task_type(part_three, task_type, choice_schema)
        return (
            "\n".join(
                [
                    part_two,
                    part_three,
                ]
            ).strip(),
            "normal:PART_THREE",
        )
    if strategy == "super_correct":
        super_correct_prompt = _adapt_prompt_for_task_type(
            prompt_module.SUPER_CORRECT_PROMPTS.strip(), task_type, choice_schema
        )
        return (super_correct_prompt, "super_correct:SUPER_CORRECT_PROMPTS")
    if strategy in STRATEGY_TO_PROMPT_NAMES:
        prompt_candidates = STRATEGY_TO_PROMPT_NAMES[strategy]
        if detailed:
            prompt_candidates = DETAILED_STRATEGY_TO_PROMPT_NAMES.get(strategy, prompt_candidates)
        chosen = None
        for candidate in prompt_candidates:
            if hasattr(prompt_module, candidate) and getattr(prompt_module, candidate, "").strip():
                chosen = candidate
                break
        part_three = _resolve_prompt_from_module(prompt_candidates)
        part_two = _adapt_prompt_for_task_type(prompt_module.PART_TWO_TASK.strip(), task_type, choice_schema)
        part_three = _adapt_prompt_for_task_type(part_three, task_type, choice_schema)
        combined = "\n".join(
            [
                part_two,
                part_three,
            ]
        ).strip()
        return (combined, f"{strategy}:{chosen or 'unknown_prompt'}")
    if strategy == "cot":
        header = (
            'Use chain-of-thought reasoning ("Let\'s think step by step") to solve the problem.\n\n'
            "Rules:\n- Finish the last answer with the numeric answer.\n"
            "- Keep reasoning concise and focused on the calculation."
        )
        if is_text_task:
            header = (
                'Use chain-of-thought reasoning ("Let\'s think step by step") to solve the problem.\n\n'
                "Rules:\n- Finish the last answer with true or false.\n"
                "- Keep reasoning concise and focused on the logic."
            )
        if choice_schema:
            choices_str = choice_schema.strip("<>")
            header = (
                'Use chain-of-thought reasoning ("Let\'s think step by step") to solve the problem.\n\n'
                f"Rules:\n- Finish the last answer with one of: {choices_str}.\n"
                "- Keep reasoning concise and focused on selecting the correct option."
            )
        tail = f"Q: {QUESTION_PLACEHOLDER}\nA:"
        body = "\n\n".join([header, tail]).strip()
        return NO_SYSTEM_PREFIX + body, "cot:zero_shot"
    if strategy == "sgft":
        sgft_body = f"{SGFT_PROMPT_IV}\n\nQ: {QUESTION_PLACEHOLDER}"
        return PLAIN_TARGET_FLAG + NO_SYSTEM_PREFIX + sgft_body, "sgft:prompt_iv"
    raise ValueError(f"Unknown strategy '{strategy}'. Options: normal, super_correct, rpb, freeform, cot, sgft")


def build_instructions(
    strategy: str,
    cot_shots: int | None = None,
    detailed: bool = False,
    task_type: str = "math",
    choice_schema: Optional[str] = None,
) -> str:
    instructions, _ = build_instructions_with_source(
        strategy, cot_shots=cot_shots, detailed=detailed, task_type=task_type, choice_schema=choice_schema
    )
    return instructions


def load_id_sequence(path: Optional[Path]) -> Optional[List[str]]:
    """Return an ordered list of IDs to keep; preserves file order."""
    if path is None:
        return None
    if not path.exists():
        print(f"Warning: intersection file {path} not found; skipping ID filtering.", flush=True)
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids or None


def build_messages(example: dict, instructions: str) -> List[Dict[str, str]]:
    plain_target = instructions.startswith(PLAIN_TARGET_FLAG)
    if plain_target:
        instructions = instructions[len(PLAIN_TARGET_FLAG):]
    use_no_system = instructions.startswith(NO_SYSTEM_PREFIX)
    stripped_instructions = instructions[len(NO_SYSTEM_PREFIX):] if use_no_system else instructions
    system = "" if use_no_system else prompt_module.PART_ONE_ROLE.strip()
    question_text = example.get("question", "").strip()
    options_text = _extract_options_text(example)
    question_text = _append_options(question_text, options_text)
    if QUESTION_PLACEHOLDER in stripped_instructions:
        user = stripped_instructions.replace(QUESTION_PLACEHOLDER, question_text)
    else:
        user = f"Question:\n{question_text}\n\n{stripped_instructions}"
    target = None
    rationale_value = example.get("response_rationale", "")
    answer_value = example.get("response_ans")
    key_order = example.get("_rationale_key_order")
    if example.get("_flatten_targets"):
        rationale_value = _coerce_plain_text(rationale_value, key_order=key_order)
        print("After flatten: "+rationale_value)
        answer_value = _clean_answer(answer_value)
    elif isinstance(rationale_value, dict):
        rationale_value = _reorder_dict(rationale_value, key_order)
    if plain_target:
        target = str(rationale_value).strip() if rationale_value else ""
    else:
        target = json.dumps({"rationale": rationale_value, "ans": answer_value}, ensure_ascii=False)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def tokenize_examples(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    instructions: str,
    pad_to_max: bool,
    flatten_targets: bool,
    print_chat: bool,
) -> dict:
    if flatten_targets:
        example = dict(example)
        example["_flatten_targets"] = True
    chat = build_messages(example, instructions)
    chat_text = format_chat(tokenizer, chat, add_generation_prompt=False)
    if print_chat:
        sample_id = example.get("index") or example.get("id") or ""
        prefix = f"[TOKENIZE] id={sample_id}" if sample_id != "" else "[TOKENIZE]"
        print(f"{prefix}\n{chat_text}\n", flush=True)
    tokenized = tokenizer(
        chat_text,
        truncation=True,
        max_length=max_length,
        padding="max_length" if pad_to_max else "longest",
    )
    return tokenized


def build_messages_for_inference(example: dict, instructions: str) -> List[Dict[str, str]]:
    if instructions.startswith(PLAIN_TARGET_FLAG):
        instructions = instructions[len(PLAIN_TARGET_FLAG):]
    use_no_system = instructions.startswith(NO_SYSTEM_PREFIX)
    stripped_instructions = instructions[len(NO_SYSTEM_PREFIX):] if use_no_system else instructions
    system = "" if use_no_system else prompt_module.PART_ONE_ROLE.strip()
    question_text = example.get("question", "").strip()
    options_text = _extract_options_text(example)
    question_text = _append_options(question_text, options_text)
    if QUESTION_PLACEHOLDER in stripped_instructions:
        user = stripped_instructions.replace(QUESTION_PLACEHOLDER, question_text)
    else:
        user = f"Question:\n{question_text}\n\n{stripped_instructions}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def format_chat(tokenizer, chat: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """Format chat text; fallback if tokenizer lacks chat_template (e.g., Llama-2)."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    system = chat[0]["content"] if chat and chat[0]["role"] == "system" else ""
    user = chat[1]["content"] if len(chat) > 1 and chat[1]["role"] == "user" else ""
    assistant = chat[2]["content"] if len(chat) > 2 and chat[2]["role"] == "assistant" else ""
    if add_generation_prompt:
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>"


def parse_answer_field(answer_text) -> tuple[str, str]:
    """Split gold answer into rationale/ans; tolerate non-string (e.g., bool)."""
    if answer_text is None:
        return "", ""
    text = str(answer_text)
    if not text:
        return "", ""
    if "####" in text:
        parts = text.split("####", 1)
        rationale = parts[0].strip()
        ans = parts[1].strip()
    else:
        rationale = text.strip()
        ans = ""
    return rationale, ans


def maybe_apply_lora(model: AutoModelForCausalLM, cfg: object):
    """Attach LoRA only when explicitly requested; full-SFT leaves model untouched."""
    use_lora = getattr(cfg, "use_lora", True)
    if not use_lora:
        print("[TRAIN] Full-SFT mode active: LoRA adapters are disabled.", flush=True)
        log_trainable_parameter_counts(model)
        return model
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA training. Install it before launching training.")
    target_modules = tuple(getattr(cfg, "lora_target_modules", DEFAULT_LORA_TARGET_MODULES)) or DEFAULT_LORA_TARGET_MODULES
    lora_config = LoraConfig(
        r=getattr(cfg, "lora_r", 0),
        lora_alpha=getattr(cfg, "lora_alpha", 0),
        lora_dropout=float(getattr(cfg, "lora_dropout", 0.0)),
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log_trainable_parameter_counts(model)
    return model


class WeightedTrainer(Trainer):
    """Trainer with per-token loss weights."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def run_distillation(cfg: DistillConfig) -> None:
    assert_uniform_training(cfg.uniform)

    rows = load_jsonl(cfg.data_file, cfg.max_samples)
    if cfg.train_size is None:
        raise ValueError("train_size must be provided so all strategies use the exact same question IDs.")
    selected_rows, train_question_ids = prepare_training_rows(rows, cfg.train_size)
    enforce_uniform_signature(cfg, train_question_ids)
    set_seed(cfg.uniform.seed)

    normalized_rows: list[dict] = []
    for row in selected_rows:
        row = dict(row)
        if cfg.strategy == "freeform":
            rat = row.get("response_rationale", row.get("rationale"))
            key_order = None
            if isinstance(rat, dict):
                key_order = list(rat.keys())
            if rat is not None:
                row["response_rationale"] = _coerce_plain_text(rat, key_order=key_order)
            ans_val = row.get("response_ans", row.get("ans"))
            if isinstance(ans_val, (dict, list)):
                row["response_ans"] = _coerce_plain_text(ans_val)
            elif ans_val is not None:
                row["response_ans"] = ans_val
            resp_payload = row.get("response_payload")
            if isinstance(resp_payload, (dict, list)):
                row["response_payload"] = json.dumps(resp_payload, ensure_ascii=False)
            elif resp_payload is not None and not isinstance(resp_payload, str):
                row["response_payload"] = str(resp_payload)
            resp_rat = row.get("response_rationale")
            if isinstance(resp_rat, (dict, list)):
                row["response_rationale"] = _coerce_plain_text(resp_rat, key_order=None)
            elif resp_rat is not None and not isinstance(resp_rat, str):
                row["response_rationale"] = str(resp_rat)
        if cfg.strategy == "sgft":
            sg_val = row.get("sg", row.get("response_rationale", ""))
            row["response_rationale"] = sg_val if sg_val else ""
            row["response_ans"] = ""
        row = _stringify_row(row)
        normalized_rows.append(row)
    raw_ds = Dataset.from_list(normalized_rows)

    choice_schema = _choice_schema_for_path(cfg.data_file)
    instructions, prompt_source = build_instructions_with_source(
        cfg.strategy,
        cot_shots=cfg.cot_shots,
        detailed=cfg.detailed,
        task_type=cfg.task_type,
        choice_schema=choice_schema,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.uniform.tokenizer_name, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer {cfg.uniform.tokenizer_name}; consistent tokenization is required."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def _tokenize(example: dict) -> dict:
        return tokenize_examples(
            example,
            tokenizer=tokenizer,
            max_length=cfg.uniform.max_length,
            instructions=instructions,
            pad_to_max=cfg.uniform.pad_to_max,
            flatten_targets=cfg.flatten_targets,
            print_chat=cfg.print_chat,
        )

    tokenized_ds = raw_ds.map(
        _tokenize,
        batched=False,
        remove_columns=raw_ds.column_names,
    )

    has_cuda = torch.cuda.is_available()
    if cfg.uniform.bf16 and not has_cuda:
        print("Warning: --bf16 requested but CUDA is unavailable. Falling back to float32.", flush=True)
    dtype = (
        torch.bfloat16
        if cfg.uniform.bf16 and has_cuda
        else torch.float16
        if has_cuda
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.uniform.model_name,
        device_map="auto",
        dtype=dtype,
    )
    model = maybe_apply_lora(model, cfg.uniform)
    log_uniform_config(cfg, prompt_source)

    use_gradient_checkpointing = has_cuda
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        print("Gradient checkpointing disabled (requires CUDA for efficiency).", flush=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.uniform.num_epochs,
        per_device_train_batch_size=cfg.uniform.batch_size,
        gradient_accumulation_steps=cfg.uniform.grad_accum,
        learning_rate=cfg.uniform.learning_rate,
        optim=cfg.uniform.optim,
        weight_decay=cfg.uniform.weight_decay,
        warmup_ratio=cfg.uniform.warmup_ratio,
        save_strategy="steps",
        logging_steps=cfg.uniform.logging_steps,
        logging_first_step=True,
        log_level="info",
        save_steps=cfg.uniform.save_steps,
        save_total_limit=2,
        bf16=cfg.uniform.bf16 and has_cuda,
        fp16=(not cfg.uniform.bf16) and has_cuda,
        gradient_checkpointing=use_gradient_checkpointing,
        dataloader_pin_memory=has_cuda,
        report_to="none",
        lr_scheduler_type=cfg.uniform.lr_scheduler_type,
        max_grad_norm=cfg.uniform.max_grad_norm,
        seed=cfg.uniform.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    resume_ckpt = None
    last_ckpt = get_last_checkpoint(str(cfg.output_dir)) if cfg.output_dir.exists() else None
    if last_ckpt:
        resume_ckpt = last_ckpt
        print(f"Resuming from checkpoint: {resume_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    if cfg.generate:
        run_generation(
            model,
            tokenizer,
            Path(cfg.test_file),
            Path(cfg.gen_output_file),
            cfg.uniform.max_length,
            cfg.max_new_tokens,
            instructions,
            cfg.max_gen_samples,
            cfg.temperature,
            cfg.do_sample,
        )


def run_generation(
    model,
    tokenizer,
    test_path: Path,
    out_path: Path,
    max_len: int,
    max_new_tokens: int,
    instructions: str,
    max_rows: Optional[int],
    temperature: float,
    do_sample: bool,
) -> None:
    model.eval()
    rows = load_jsonl(test_path, limit=max_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    pad_id = tokenizer.pad_token_id or eos_id

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=1.0,
        eos_token_id=eos_arg,
        pad_token_id=pad_id,
        min_new_tokens=1,
    )

    completed_ids: set[str] = set()
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as existing:
                for line in existing:
                    try:
                        obj = json.loads(line)
                        if "index" in obj:
                            completed_ids.add(str(obj["index"]))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    with out_path.open("a", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            row_id = str(row.get("index", f"test_{idx:05d}"))
            if row_id in completed_ids:
                continue
            messages = build_messages_for_inference(row, instructions)
            prompt_text = format_chat(tokenizer, messages, add_generation_prompt=True)
            print(f"\n[GEN] index={row_id}")
            print("---- Prompt ----")
            print(prompt_text)
            inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=max_len
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, **generation_kwargs)
            gen_text = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            if not gen_text.strip():
                gen_text = tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=False,
                ).strip()
            print("---- Model response ----")
            print(gen_text.strip())
            gold_ans = None
            for key in ("answer", "answer_from_dataset", "gold_ans", "gold", "ans"):
                if key in row:
                    val = row.get(key)
                    gold_ans = val
                    break
            if isinstance(gold_ans, str):
                _, parsed_ans = parse_answer_field(gold_ans)
                if parsed_ans not in ("", None):
                    gold_ans = parsed_ans

            model_ans = None
            try:
                payload = json.loads(gen_text)
                if isinstance(payload, dict):
                    model_ans = payload.get("ans")
            except Exception:
                model_ans = None

            answer_value = row.get("answer")
            if answer_value in (None, ""):
                answer_value = gold_ans

            out_row = dict(row)
            out_row.update(
                {
                    "index": row_id,
                    "answer": answer_value,
                    "gold_ans": gold_ans,
                    "model_response": gen_text.strip(),
                    "model_ans": model_ans,
                }
            )
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    strategy = args.strategy.lower()
    valid_structured = set(STRATEGY_TO_DIR.keys())
    if strategy not in {"normal", "super_correct", "freeform", "cot", "sgft"} and strategy not in valid_structured:
        raise SystemExit(f"Unknown strategy '{strategy}'. Options: normal, super_correct, freeform, cot, sgft, {list(valid_structured)}")

    if args.data_file:
        data_file = args.data_file
    else:
        if strategy == "normal":
            data_file = NORMAL_DATA_FILE
        elif strategy == "super_correct":
            data_file = SUPER_CORRECT_DATA_FILE
        elif strategy in STRATEGY_TO_DIR:
            dir_name = STRATEGY_TO_DIR[strategy]
            data_file = STRUCT_BASE_DIR / f"filtered_{dir_name}.jsonl"
        else:
            raise SystemExit(f"No default dataset for strategy '{strategy}'. Please provide --data-file.")

    task_type = _normalize_task_type(args.task_type, data_file)
    if args.task_type is None and task_type != "math":
        print(f"[PROMPT] Auto-detected task_type='{task_type}' from data path.", flush=True)

    model_slug = slugify(args.model_name)
    scratch_root = REPO_ROOT / "outputs" / "checkpoints"
    strategy_label = strategy
    default_output = scratch_root / f"{strategy_label}_{model_slug}"
    output_dir = args.output_dir or default_output
    intersection_file = args.intersection_file if strategy == "rpb" else None
    dataset_label = (args.test_file or Path("test")).stem
    size_label = args.train_size if args.train_size is not None else "all"
    default_pred = output_dir / f"{strategy_label}_{model_slug}_{dataset_label}_{size_label}.jsonl"
    gen_output_file = args.gen_output_file if args.gen_output_file else default_pred

    uniform_cfg = UniformTrainingConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name or args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim=args.optim,
        bf16=args.bf16,
        use_lora=False,
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_target_modules=tuple(),
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        pad_to_max=args.pad_to_max_length,
    )

    signature_path = args.uniform_signature_file.resolve()

    cfg = DistillConfig(
        data_file=data_file,
        output_dir=output_dir,
        strategy=strategy,
        task_type=task_type,
        detailed=args.detailed,
        intersection_file=intersection_file,
        max_samples=args.max_samples,
        train_size=args.train_size,
        uniform=uniform_cfg,
        cot_shots=args.cot_shots,
        generate=args.generate,
        test_file=args.test_file,
        gen_output_file=gen_output_file,
        max_gen_samples=args.max_gen_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        flatten_targets=args.flatten_targets,
        print_chat=args.print_chat,
        signature_file=signature_path,
        reset_signature=args.reset_uniform_signature,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_distillation(cfg)


if __name__ == "__main__":
    main()
