from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
import time
import warnings
from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from drpc.teacher import prompts as prompt_defaults

    DEFAULT_NORMAL_SHOTS = getattr(prompt_defaults, "DEFAULT_NORMAL_FEW_SHOT_COUNT", 0)
except Exception:
    DEFAULT_NORMAL_SHOTS = 0

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data/GSM8K/train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "structure_rationale"
DEFAULT_STRATEGY = "normal"

TASK_TYPE_HINTS = {
    "strategyqa": "text",
    "strategy_qa": "text",
    "commonsense": "text",
    "common": "text",
    "ai2arc": "text",
    "gpqa": "text",
    "svamp": "math",
    "gsm8k": "math",
    "aqua": "math",
}

DATASET_PRESETS = {
    "gsm8k": {
        "path": PROJECT_ROOT / "data" / "GSM8K" / "train.jsonl",
        "task_type": "math",
    },
    "svamp": {
        "path": PROJECT_ROOT / "data" / "SVAMP" / "SVAMP.json",
        "task_type": "math",
    },
    "strategyqa": {
        "path": PROJECT_ROOT / "data" / "StrategyQA" / "train.json",
        "task_type": "commonsense",
        "task_instruction": (
            "You are solving a commonsense, causal, or logical reasoning task.\n"
            "- Decompose the reasoning into abstract steps (e.g., RetrieveFact, InferConsequence).\n"
            "- Each step should express the reasoning explicitly and reflects inference or logic."
        ),
    },
}

CATEGORY_BASED_FREEFORM_PROMPT_NAMES: Sequence[str] = (
    "Structured_CATEGORY_BASED_FREEFORM_PART_THREE",
    "STRUCTURED_CATEGORY_BASED_FREEFORM_PART_THREE",
)
CATEGORY_BASED_FREEFORM_DETAILED_PROMPT_NAMES: Sequence[str] = (
    "Structured_CATEGORY_BASED_FREEFORM_PART_THREE_DETAILED",
    "STRUCTURED_CATEGORY_BASED_FREEFORM_PART_THREE_DETAILED",
)

PROMPT_STRATEGIES = {
    "normal": "normal",
    "freeform": "freeform",
    "rpb": "rpb",
    "super_correct": "super_correct",
    "cot": "cot",
    "sgft": "sgft",
}

ID_FIELDS: Sequence[str] = ("id", "question_id", "sample_id")
SANITIZE_RE = re.compile(r"[^0-9A-Za-z_.-]+")
ANSWER_TAG_RE = re.compile(
    r"(?:<|\u27e8)\s*answer\s*(?:>|\u27e9)(.*?)(?:<|\u27e8)\s*/\s*answer\s*(?:>|\u27e9)",
    re.IGNORECASE | re.DOTALL,
)
ANSWER_CHOICE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"\banswer\b\s*[:\-]?\s*(?:\*\*|__|`+)?\s*\(?\s*([A-E])\s*\)?(?:\*\*|__|`+)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcorrect\s+(?:answer|choice|pair)\s+is\s*(?:\*\*|__|`+)?\s*\(?\s*([A-E])\s*\)?(?:\*\*|__|`+)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthe\s+correct\s+(?:answer|choice|pair)\s+is\s*(?:\*\*|__|`+)?\s*\(?\s*([A-E])\s*\)?(?:\*\*|__|`+)?\b",
        re.IGNORECASE,
    ),
)
OPTION_CHOICE_RE = re.compile(
    r"\boption\b\s*(?:\*\*|__|`+)?\s*\(?\s*([A-E])\s*\)?(?:\*\*|__|`+)?\b",
    re.IGNORECASE,
)
OPTION_LINE_RE = re.compile(r"(?:^|\n)\s*([A-E])\s*[\)\.\:\-]\s*([^\n\r]+)", re.IGNORECASE)
ANS_FIELD_RE = re.compile(
    r"(?i)(?:\"ans\"|'ans'|\bans\b)\s*:\s*(\"([^\"]*)\"|'([^']*)'|([^,}\n]+))"
)
FREEFORM_PROMPT_NAMES: Sequence[str] = (
    "Structured_RPB_PART_THREE",
    "STRUCTURED_FREEFORM_PART_THREE",
    "STRUCTURED_FREE_FORM_PART_THREE",
    "SLM_PART_THREE_PROMPTS",
)
FREEFORM_DETAILED_PROMPT_NAMES: Sequence[str] = (
    "Structured_RPB_PART_THREE_DETAILED",
    "STRUCTURED_FREEFORM_PART_THREE_DETAILED",
    "STRUCTURED_FREE_FORM_PART_THREE_DETAILED",
)
GENERIC_FREEFORM_PROMPT_NAMES: Sequence[str] = (
    "FREEFORM_REASONING_PATH_PART_THREE",
)
GENERIC_FREEFORM_DETAILED_PROMPT_NAMES: Sequence[str] = (
    "FREEFORM_REASONING_PATH_PART_THREE_DETAILED",
)

CHOICE_ANSWER_SCHEMA = "<A|B|C|D|E>"


def _extract_options_text(sample: dict) -> str:
    """Return the options/choices block if present on the sample."""
    for key in ("options", "Options", "option", "Option"):
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _append_options(question_text: str, options_text: str) -> str:
    """Append options to the question text if not already present."""
    if not options_text:
        return question_text.rstrip()
    base = question_text.rstrip()
    if options_text in base:
        return base
    joiner = "" if base.endswith("\n") else "\n"
    return f"{base}{joiner}{options_text}"


def _append_choice_instruction(question_text: str, choices_schema: str) -> str:
    """Add an explicit answer-choice instruction line if not present."""
    choices = [c.strip() for c in choices_schema.strip("<>").split("|") if c.strip()]
    if not choices:
        return question_text
    if len(choices) == 1:
        choice_line = f"Answer with: {choices[0]}."
    else:
        choice_line = f"Answer with one of: {', '.join(choices[:-1])}, or {choices[-1]}."
    if choice_line in question_text:
        return question_text
    joiner = "" if question_text.rstrip().endswith("\n") else "\n"
    return f"{question_text.rstrip()}{joiner}{choice_line}"


def _infer_choice_answer_schema(
    sample: dict | None = None,
    options_text: str | None = None,
    dataset_name: str | None = None,
    dataset_path: Any | None = None,
) -> str | None:
    """Infer a multiple-choice answer schema from option text or known dataset names."""
    options_block = options_text if options_text is not None else _extract_options_text(sample or {})
    option_map = _extract_option_map(options_block)
    labels = [label for label in "ABCDE" if label in option_map]
    if len(labels) >= 2:
        return "<" + "|".join(labels) + ">"

    identifier_text = " ".join(
        part.strip().lower()
        for part in (str(dataset_name or ""), str(dataset_path or ""))
        if str(part or "").strip()
    )
    if any(name in identifier_text for name in ("aqua", "ai2arc", "gpqa")):
        return CHOICE_ANSWER_SCHEMA
    return None


def _extract_option_map(options_text: str) -> Dict[str, str]:
    """Parse multi-choice options into a letter->value map."""
    if not isinstance(options_text, str) or not options_text.strip():
        return {}
    out: Dict[str, str] = {}
    for letter, option_value in OPTION_LINE_RE.findall(options_text):
        out[letter.upper()] = option_value.strip()
    return out


def _is_choice_letter(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) == 1 and value.strip().upper() in {"A", "B", "C", "D", "E"}


def _extract_choice_from_text(text: Any) -> str | None:
    """Extract an A-E answer marker from text."""
    if not isinstance(text, str):
        return None
    tag_match = ANSWER_TAG_RE.search(text)
    if tag_match:
        token = _normalize_answer_token(tag_match.group(1))
        if _is_choice_letter(token):
            return str(token).upper()
    ans_match = ANS_FIELD_RE.search(text)
    if ans_match:
        token = ans_match.group(2) or ans_match.group(3) or ans_match.group(4) or ""
        normalized = _normalize_answer_token(token)
        if _is_choice_letter(normalized):
            return str(normalized).upper()
    for pattern in ANSWER_CHOICE_PATTERNS:
        choice_match = pattern.search(text)
        if choice_match:
            return choice_match.group(1).upper()
    option_match = OPTION_CHOICE_RE.search(text)
    if option_match:
        return option_match.group(1).upper()
    return None


def _map_answer_to_choice(answer: Any, options_text: str) -> Any:
    """Map numeric/text answers to A-E using option values when available."""
    if answer is None:
        return None

    if isinstance(answer, str):
        cleaned = answer.strip().strip("\"'")
        if _is_choice_letter(cleaned):
            return cleaned.upper()
        letter_match = re.search(r"\b([A-E])\b", cleaned, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

    option_map = _extract_option_map(options_text)
    if not option_map:
        return answer

    matches: List[str] = []
    for letter, option_value in option_map.items():
        if _answers_match(option_value, answer):
            matches.append(letter)
    if len(matches) == 1:
        return matches[0]

    answer_num = _coerce_number(answer)
    if isinstance(answer_num, (int, float)) and not isinstance(answer_num, bool):
        numeric_options: List[Tuple[str, float]] = []
        for letter, option_value in option_map.items():
            opt_num = _coerce_number(option_value)
            if isinstance(opt_num, (int, float)) and not isinstance(opt_num, bool):
                numeric_options.append((letter, float(opt_num)))
        if numeric_options:
            numeric_options.sort(key=lambda item: abs(item[1] - float(answer_num)))
            if len(numeric_options) == 1:
                return numeric_options[0][0]
            best_letter, best_value = numeric_options[0]
            second_value = numeric_options[1][1]
            best_diff = abs(best_value - float(answer_num))
            second_diff = abs(second_value - float(answer_num))
            if best_diff < second_diff:
                return best_letter
    return answer


def _apply_answer_schema(prompt_text: str, choices_schema: str | None) -> str:
    """Replace numeric/bool placeholders with multiple-choice schema."""
    if not prompt_text or not choices_schema:
        return prompt_text
    choices = [c.strip() for c in choices_schema.strip("<>").split("|") if c.strip()]
    if len(choices) > 1:
        choices_str = ", ".join(choices[:-1]) + f", or {choices[-1]}"
    elif choices:
        choices_str = choices[0]
    else:
        choices_str = choices_schema
    choice_rule = f'"ans" must be one of {choices_str}.'
    explicit_rule = None
    if len(choices) >= 2 and all(len(choice) == 1 and choice.isalpha() for choice in choices):
        if len(choices) == 2:
            choices_text = f"{choices[0]} or {choices[1]}"
        else:
            choices_text = ", ".join(choices[:-1]) + f", or {choices[-1]}"
        explicit_rule = f'- "ans" must be one of {choices_text} (letter only, not a numeric value).'
        choice_rule = explicit_rule[2:]
    replacements = {
        "<numeric>": choices_schema,
        "<bool>": choices_schema,
        '"ans" must be numeric (no strings, units, or words).': choice_rule,
        '"ans" must be true or false.': choice_rule,
    }
    new_text = prompt_text
    for old, new in replacements.items():
        new_text = new_text.replace(old, new)
    return new_text


def apply_dataset_preset(args) -> None:
    """
    Apply dataset preset overrides to argparse args in-place.

    - If --dataset-name matches a preset, set dataset_path and task_type accordingly.
    - Otherwise, if dataset_path hints at StrategyQA, default task_type to text.
    - Otherwise, infer task_type from TASK_TYPE_HINTS using dataset name/path.
    - If dataset_path hints at StrategyQA and no dataset_name is set, set dataset_name to "strategyqa".
    """
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name:
        preset = DATASET_PRESETS.get(dataset_name)
        if preset:
            args.dataset_path = Path(preset["path"])
            preset_task = preset.get("task_type", getattr(args, "task_type", None))
            args.task_type = "text" if preset_task in {"commonsense", "common"} else preset_task
            return

    if dataset_name:
        hint_key = dataset_name.lower()
        hinted = next((v for k, v in TASK_TYPE_HINTS.items() if k in hint_key), None)
        if hinted:
            args.task_type = hinted

    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        path_text = str(dataset_path).lower()
        hinted = next((v for k, v in TASK_TYPE_HINTS.items() if k in path_text), None)
        if hinted:
            args.task_type = hinted
        if "strategyqa" in path_text and not getattr(args, "dataset_name", None):
            args.dataset_name = "strategyqa"
    if dataset_path and "strategyqa" in str(dataset_path).lower():
        args.task_type = "text"
        if not getattr(args, "dataset_name", None):
            args.dataset_name = "strategyqa"


def set_global_seed(seed: int = 42) -> None:
    """Best-effort global seeding for reproducibility across libs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch as th
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_strategy(name: str | None, default: str = DEFAULT_STRATEGY) -> str:
    """Normalize a strategy string to a known key."""
    candidate = default if not name else name.strip().lower()
    if candidate in PROMPT_STRATEGIES:
        return candidate
    valid = ", ".join(sorted(PROMPT_STRATEGIES))
    raise SystemExit(f"Unknown prompt strategy '{name}'. Valid options: {valid}")


def _resolve_prompt_from_module(prompt_module, candidate_names: Sequence[str]) -> Tuple[str, str]:
    """Return the first matching prompt (name, text) from the module."""
    for name in candidate_names:
        if not hasattr(prompt_module, name):
            continue
        prompt_obj = getattr(prompt_module, name)
        if callable(prompt_obj):
            try:
                prompt_text = prompt_obj()
            except TypeError:
                prompt_text = prompt_obj
        else:
            prompt_text = prompt_obj
        if isinstance(prompt_text, str) and prompt_text.strip():
            return name, prompt_text.strip()
    searched = ", ".join(candidate_names)
    raise SystemExit(f"No prompt found in prompts module. Looked for: {searched}")


def _format_few_shot_examples(examples: Sequence[Tuple[str, str]], shot_count: int) -> str:
    """Format few-shot examples as labeled blocks."""
    selected = list(examples)[:shot_count]
    blocks = []
    for idx, (question, answer) in enumerate(selected, start=1):
        blocks.append(f"Example {idx}:\nQuestion: {question}\nAnswer:\n{answer}")
    return "\n\n".join(blocks)


def _build_normal_part_three(prompt_module, normal_shots: int | None) -> Tuple[str, str]:
    """Construct the part-three prompt for the normal strategy."""
    if hasattr(prompt_module, "normalize_part_three"):
        try:
            part_three = prompt_module.normalize_part_three(normal_shots)
            if isinstance(part_three, str) and part_three.strip():
                return "NORMAL_PART_THREE", part_three.strip()
        except Exception:
            pass
    examples = getattr(prompt_module, "GSM8K_FEW_SHOT_EXAMPLES", [])
    if not examples:
        raise SystemExit("Normal strategy requires GSM8K_FEW_SHOT_EXAMPLES in prompts.py.")

    total_available = len(examples)
    shot_limit = total_available if normal_shots is None else int(normal_shots)
    shot_count = max(0, min(shot_limit, total_available))
    if shot_count <= 0:
        raise SystemExit("Normal strategy needs at least one few-shot example (set --normal-shots).")

    formatted_examples = _format_few_shot_examples(examples, shot_count)
    prompt_body = (
        "Follow the JSON format shown in these examples.\n\n"
        f"{formatted_examples}\n\n"
        "Return ONLY valid JSON for the new question using the same structure."
    )
    return "NORMAL_FEW_SHOT_PART_THREE", prompt_body.strip()


def _extract_final_number(text: str) -> str:
    if not isinstance(text, str):
        return ""
    candidate = text.strip().split()[-1].strip(",. ")
    return candidate


def _build_cot_part_three(prompt_module, cot_shots: int | None) -> Tuple[str, str]:
    """Construct the few-shot examples block for the CoT strategy using GSM8K examples."""
    examples = getattr(prompt_module, "GSM8K_FEW_SHOT_EXAMPLES", [])
    total_available = len(examples)
    shot_limit = total_available if cot_shots is None else int(cot_shots)
    shot_count = max(0, min(shot_limit, total_available))

    if shot_count <= 0:
        zero_shot = "Let's think step by step. Show concise reasoning and end with Answer."
        return "COT_ZERO_SHOT_PART_THREE", zero_shot.strip()

    selected = examples[:shot_count]
    blocks = [f"Question: {q}\nAnswer: {a}" for q, a in selected]
    prompt_body = "\n\n".join(blocks)
    return "COT_FEW_SHOT_PART_THREE", prompt_body.strip()


def _sgft_prompt_iv(prompt_module) -> str:
    """
    Return the SGFT guide prompt (prompt_iv) used by Baselines/SGFT.

    Source wording mirrors:
    - Baselines/SGFT/infer/collab_infer.py::PROMPT_IV
    - drpc/student/distill_rationale.py::SGFT_PROMPT_IV
    """
    candidate_names = ("SGFT_PROMPT_IV", "PROMPT_IV")
    for name in candidate_names:
        value = getattr(prompt_module, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return (
        "Please generate a step-by-step solution for the following problem with no calculations.\n"
        " You don't need to solve it, just output the steps in 2 to 6 steps."
    )


def build_common_arg_parser(description: str, default_model: str) -> argparse.ArgumentParser:
    """Create an argument parser shared by query scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-name",
        type=str.lower,
        choices=sorted(DATASET_PRESETS),
        help="Preset dataset to use (overrides --dataset-path). Choices: gsm8k, svamp, strategyqa.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the GSM8K jsonl file (default: {DEFAULT_DATASET_PATH}).",
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
        "--strategy",
        type=str.lower,
        default=DEFAULT_STRATEGY,
        choices=sorted(PROMPT_STRATEGIES),
        help="Prompt strategy to use (normal, cot, freeform, rpb, super_correct, sgft).",
    )
    parser.add_argument(
        "--category-based",
        action="store_true",
        help="Use category/intents from dataset (RPB only) with category-based structured prompt.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use detailed rationale prompts when available (RPB/freeform only).",
    )
    parser.add_argument(
        "--normal-shots",
        type=int,
        default=DEFAULT_NORMAL_SHOTS,
        help=(
            "Number of few-shot examples to include when strategy=normal "
            f"(default: {DEFAULT_NORMAL_SHOTS}; minimum 1, capped to available examples)."
        ),
    )
    parser.add_argument(
        "--cot-shots",
        type=int,
        default=0,
        help="Number of few-shot examples to include when strategy=cot (default: 0; capped to available examples).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model identifier to query (default: {default_model}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store responses (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature (default: use script/model default).",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling for generation (default: use script/model default).",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling for generation (default: use script/model default).",
    )
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="Print the samples and prompt but do not call the model.",
    )
    parser.add_argument(
        "--include-answer-in-prompt",
        action="store_true",
        help="Include the gold answer in the prompt (where supported, e.g., RPB/freeform wrappers).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="math",
        choices=["math", "common", "commonsense", "text"],
        help="Task type hint for prompt wrappers (math, common/commonsense, or text).",
    )

    parser.set_defaults(do_sample=None)
    return parser


def _entry_key(entry: Dict[str, Any]) -> str:
    return str(entry.get("index", ""))


def load_existing_results(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    entries: List[Dict[str, Any]] = []
    processed_ids: set = set()
    if not path.exists():
        return entries, processed_ids
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    entries.append(obj)
                    idx = obj.get("index")
                    if idx is not None:
                        processed_ids.add(str(idx))
    except Exception as exc:
        print(f"Warning: Could not read {path}: {exc}")
    entries.sort(key=_entry_key)
    return entries, processed_ids


def write_results(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Rewrite the JSONL file with the provided entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in entries:
            handle.write(json.dumps(_make_json_safe(record), ensure_ascii=False))
            handle.write("\n")


def _make_json_safe(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_make_json_safe(v) for v in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    return str(value)


FRACTION_TEXT_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$")
LATEX_FRAC_BRACED_PATTERN = re.compile(r"^\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}$")
LATEX_FRAC_COMPACT_PATTERN = re.compile(r"^\\frac([+-]?\d+(?:\.\d+)?)([+-]?\d+(?:\.\d+)?)$")


def _parse_float_token(token: str) -> Any:
    t = str(token).strip().replace(",", "").replace("−", "-")
    if not t:
        return None
    try:
        return float(t)
    except Exception:
        return None


def _parse_fraction_value(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    candidate = text.strip().replace("−", "-")
    if not candidate:
        return None

    sign = 1.0
    if candidate[0] in "+-":
        sign = -1.0 if candidate[0] == "-" else 1.0
        candidate = candidate[1:].strip()
        if not candidate:
            return None

    match = LATEX_FRAC_BRACED_PATTERN.fullmatch(candidate)
    if match:
        num = _parse_float_token(match.group(1))
        den = _parse_float_token(match.group(2))
        if num is None or den in (None, 0.0):
            return None
        value = sign * (float(num) / float(den))
        if math.isfinite(value):
            return value
        return None

    compact = candidate.replace(" ", "")
    match = LATEX_FRAC_COMPACT_PATTERN.fullmatch(compact)
    if match:
        num = _parse_float_token(match.group(1))
        den = _parse_float_token(match.group(2))
        if num is None or den in (None, 0.0):
            return None
        value = sign * (float(num) / float(den))
        if math.isfinite(value):
            return value
        return None

    match = FRACTION_TEXT_PATTERN.fullmatch(candidate)
    if match:
        num = _parse_float_token(match.group(1))
        den = _parse_float_token(match.group(2))
        if num is None or den in (None, 0.0):
            return None
        value = sign * (float(num) / float(den))
        if math.isfinite(value):
            return value
        return None
    return None


def _coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    if isinstance(value, str):
        candidate = value.strip().replace(",", "")
        candidate = candidate.replace("−", "-")
        candidate = candidate.strip("$₹€£")
        candidate = candidate.rstrip("%")
        frac = _parse_fraction_value(candidate)
        if frac is not None:
            if abs(frac - round(frac)) < 1e-12:
                return int(round(frac))
            return frac
        try:
            if candidate:
                num = float(candidate)
                if num.is_integer():
                    return int(num)
                return num
        except ValueError:
            wrapped = re.fullmatch(r"[^\d\-+]*([\-+]?\d+(?:\.\d+)?)[^\d]*", candidate)
            if wrapped:
                try:
                    num = float(wrapped.group(1))
                    if num.is_integer():
                        return int(num)
                    return num
                except ValueError:
                    return value
            return value
    return value


def _normalize_answer_token(token: Any) -> Any:
    """Normalize a raw answer token into choice letter, number, or bool when possible."""
    if not isinstance(token, str):
        return _coerce_number(token)
    cleaned = token.strip().strip("\"'").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) == 1 and cleaned.upper() in {"A", "B", "C", "D", "E"}:
        return cleaned.upper()
    letter_match = re.search(r"\b([A-E])\b", cleaned, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    return _coerce_number(cleaned)


def _answers_match(gold: Any, predicted: Any) -> bool:
    """Heuristic answer equality supporting numeric and bool normalization."""
    if gold is None or predicted is None:
        return False
    gold_num = _coerce_number(gold)
    pred_num = _coerce_number(predicted)

    if isinstance(gold_num, bool) and isinstance(pred_num, bool):
        return gold_num is pred_num

    if isinstance(gold_num, (int, float)) and not isinstance(gold_num, bool):
        if isinstance(pred_num, (int, float)) and not isinstance(pred_num, bool):
            return abs(float(gold_num) - float(pred_num)) < 1e-9

    gold_str = str(gold).strip().lower()
    pred_str = str(predicted).strip().lower()
    return gold_str == pred_str


def load_samples(dataset_path: Path, start_index: int, num_samples: int) -> List[Tuple[int, dict]]:
    """Load a consecutive slice of samples (jsonl or json array; stops at EOF)."""
    if start_index < 0 or num_samples <= 0:
        raise ValueError("sample index must be >= 0 and num-samples must be positive.")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {dataset_path}. "
            "Use --dataset-path to point to a jsonl file."
        )

    collected: List[Tuple[int, dict]] = []
    text = dataset_path.read_text(encoding="utf-8")
    parsed_records: List[dict] | None = None
    try:
        parsed_obj = json.loads(text)
        if isinstance(parsed_obj, list):
            parsed_records = [row for row in parsed_obj if isinstance(row, dict)]
        elif isinstance(parsed_obj, dict):
            for key in ("data", "examples"):
                if key in parsed_obj and isinstance(parsed_obj[key], list):
                    parsed_records = [row for row in parsed_obj[key] if isinstance(row, dict)]
                    break
            if parsed_records is None:
                parsed_records = [parsed_obj]
    except Exception:
        parsed_records = None

    if parsed_records is not None:
        for line_idx, sample in enumerate(parsed_records):
            if line_idx < start_index:
                continue
            collected.append((line_idx, sample))
            if len(collected) >= num_samples:
                break
    else:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_idx, raw_line in enumerate(handle):
                normalized_line = raw_line.lstrip("\ufeff")
                cleaned_line = re.sub(r"[\uFEFF\u200B\u200C\u200D\u2060\x00]", "", normalized_line).strip()
                if line_idx < start_index or not cleaned_line:
                    continue
                try:
                    sample = json.loads(cleaned_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSON on line {line_idx + 1} of {dataset_path}: {exc}"
                    ) from exc
                collected.append((line_idx, sample))
                if len(collected) >= num_samples:
                    break

    if len(collected) < num_samples:
        print(
            f"Warning: Requested {num_samples} samples starting at index {start_index}, "
            f"but only {len(collected)} were available in {dataset_path}."
        )
    return collected


def _apply_dataset_normalization(sample: dict, dataset_name: str | None) -> dict:
    """Harmonize sample fields (question/answer/id) across datasets."""
    if not isinstance(sample, dict):
        return {"question": str(sample)}

    if dataset_name == "svamp":
        body = str(sample.get("Body", "")).strip()
        question = str(sample.get("Question", "")).strip()
        combined_question = "\n".join(part for part in (body, question) if part)
        normalized = dict(sample)
        normalized["question"] = combined_question
        normalized["answer"] = str(sample.get("Answer", "")).strip()
        normalized.setdefault("id", sample.get("ID"))
        return normalized

    if dataset_name == "strategyqa":
        normalized = dict(sample)
        normalized["question"] = str(sample.get("question", "")).strip()
        ans = sample.get("answer", "")
        normalized["answer"] = ans if isinstance(ans, bool) else str(ans).strip()
        normalized.setdefault("id", sample.get("qid"))
        preset = DATASET_PRESETS.get("strategyqa", {})
        task_instruction = preset.get("task_instruction", "")
        if task_instruction:
            normalized["_task_instruction"] = task_instruction
        return normalized

    normalized = dict(sample)
    normalized["question"] = str(sample.get("question", "")).strip()
    normalized["answer"] = str(sample.get("answer", "")).strip()
    normalized.setdefault("id", sample.get("qid"))
    return normalized


def pick_part_three_prompt(
    strategy: str,
    prompt_module,
    normal_shots: int | None = None,
    cot_shots: int | None = None,
    category_based: bool = False,
    detailed: bool = False,
) -> Tuple[str, str]:
    """Fetch the part-three instructions for the requested strategy."""
    normalized = normalize_strategy(strategy, default=DEFAULT_STRATEGY)
    if normalized == "rpb":
        if category_based:
            candidate_names = (
                CATEGORY_BASED_FREEFORM_DETAILED_PROMPT_NAMES
                if detailed
                else CATEGORY_BASED_FREEFORM_PROMPT_NAMES
            )
            return _resolve_prompt_from_module(prompt_module, candidate_names)
        candidate_names = FREEFORM_DETAILED_PROMPT_NAMES if detailed else FREEFORM_PROMPT_NAMES
        return _resolve_prompt_from_module(prompt_module, candidate_names)
    if normalized == "freeform":
        candidate_names = (
            GENERIC_FREEFORM_DETAILED_PROMPT_NAMES if detailed else GENERIC_FREEFORM_PROMPT_NAMES
        )
        return _resolve_prompt_from_module(prompt_module, candidate_names)
    if normalized == "cot":
        return _build_cot_part_three(prompt_module, cot_shots)
    return _build_normal_part_three(prompt_module, normal_shots)


def build_prompt_texts(
    system_prompt: str | None, user_prompt: str, question: str, use_system_default: bool = True
) -> Tuple[str, str]:
    """Create labeled SYSTEM/USER prompt strings."""
    stripped_system = ""
    if isinstance(system_prompt, str):
        stripped_system = system_prompt.strip()
    if use_system_default and not stripped_system:
        stripped_system = "You are a helpful math tutor."
    stripped_user = user_prompt.strip()
    question_text = question.strip()

    labeled_system = f"\n{stripped_system}" if stripped_system else ""

    user_sections = [""]
    if question_text:
        user_sections.append(f"Question:\n{question_text}")
    if stripped_user:
        user_sections.append(stripped_user)

    labeled_user = "\n\n".join(user_sections)
    return labeled_system, labeled_user


def build_prompt_bundle(
    prompt_module,
    strategy: str,
    normal_shots: int | None,
    cot_shots: int | None,
    include_teacher_requirements: bool = False,
    include_gold_answer: bool = False,
    task_type: str = "math",
    category_based: bool = False,
    detailed: bool = False,
) -> Tuple[str, Callable[[str, str | None, str, dict | None], Tuple[str, bool]]]:
    """Assemble system prompt and a builder for the user prompt.

    Returns:
        system_prompt: The system role string.
        user_prompt_builder: Callable(question, gold_answer, task_type) -> (prompt_text, includes_question)
    """
    normalized = normalize_strategy(strategy, default=DEFAULT_STRATEGY)
    base_system_prompt = getattr(
        prompt_module,
        "PART_ONE_ROLE",
        "You are a rigorous but concise math tutor.",
    )
    system_prompt = str(base_system_prompt or "").strip() or "You are a rigorous but concise math tutor."
    if normalized == "normal":
        system_prompt = ""
    if normalized == "cot":
        system_prompt = ""
    text_task_types = {"text", "commonsense", "common", "strategyqa", "strategy_qa"}

    if normalized == "super_correct":
        super_prompt = str(getattr(prompt_module, "SUPER_CORRECT_PROMPTS", "") or "").strip()
        if not super_prompt:
            raise SystemExit("super_correct strategy requires SUPER_CORRECT_PROMPTS in prompts.py.")

        def user_prompt_builder(
            question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
        ) -> Tuple[str, bool]:
            question_text = question.strip()
            if question_text:
                prompt_text = f"{super_prompt}\n\n{question_text}"
            else:
                prompt_text = super_prompt
            return prompt_text, True

        return system_prompt, user_prompt_builder

    if normalized == "sgft":
        sgft_prompt = _sgft_prompt_iv(prompt_module)
        system_prompt = ""

        def user_prompt_builder(
            question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
        ) -> Tuple[str, bool]:
            question_text = question.strip()
            if question_text:
                prompt_text = f"{sgft_prompt}\n\n{question_text}"
            else:
                prompt_text = sgft_prompt
            return prompt_text, True

        return system_prompt, user_prompt_builder

    part_two_prompt = (
        ""
        if normalized in {"normal", "cot"}
        else str(getattr(prompt_module, "PART_TWO_TASK", "") or "").strip()
    )
    _, part_three_prompt = pick_part_three_prompt(
        normalized,
        prompt_module,
        normal_shots=normal_shots,
        cot_shots=cot_shots,
        category_based=category_based,
        detailed=detailed,
    )

    if normalized == "rpb":
        if task_type in {"common", "commonsense", "text"} and not category_based:
            system_prompt = ""

            def user_prompt_builder(
                question: str, gold_answer: str | None, _: str = "text", sample_meta: dict | None = None
            ) -> Tuple[str, bool]:
                prompt_text = (
                    "You are solving a commonsense reasoning task.\n"
                    "- Decompose the reasoning into abstract steps.\n"
                    "- Each step should express the reasoning explicitly and reflects inference or logic.\n\n"
                    f"Question: {question}\n\n"
                    "Output ONLY valid JSON in this format, and AVOID QUESTION SPECIFIC WORDING in reasoning_path keys:\n"
                    "{\n"
                    '  "route": {\n'
                    '    "category": "<High level type of question>",\n'
                    '    "intent": ["<Goal 1>", "<Goal 2>"],\n'
                    '    "difficulty": <1|2|3>,\n'
                    '    "budget": <Follow Budget Contract>,\n'
                    '    "reasoning_path": ["<DescriptiveStepName1>", "<DescriptiveStepName2>", ...]\n'
                    "  },\n"
                    '  "rationale": {\n'
                    '    "<DescriptiveStepName1>": "<short reasoning with derivation>",\n'
                    '    "<DescriptiveStepName2>": "<short reasoning with derivation>",\n'
                    "    ...\n"
                    "  },\n"
                    '  "ans": <Ture or False>\n'
                    "}\n"
                    "- Category: The high level type of the question.\n"
                    "- Intent: The high level goal of the question.\n"
                    "- Difficulty: How hard is this question (1-3).\n"
                    "- Budget: The maximum number of steps allowed.\n\n"
                    "Hard rules:\n"
                    '- Output must match the required JSON schema exactly.\n'
                    '- "route" must be a dictionary containing exactly: category, intent, difficulty, budget, and reasoning_path.\n'
                    '- "reasoning_path" must be a list of strings representing the ordered steps.\n'
                    '- The keys in "rationale" must match the strings in "reasoning_path" exactly.\n\n'
                    "Budget contract:\n"
                    "- Difficulty 1 implies Budget 2.\n"
                    "- Difficulty 2 or 3 implies Budget 3.\n\n"
                    "Field-count contract:\n"
                    "- The length of the \"reasoning_path\" list must be NO MORE THAN the <budget>.\n\n"
                    "Reasoning key naming policy (CRITICAL):\n"
                    "1. Format: Keys (in reasoning_path and rationale) must be TitleCase letters only (A-Z, a-z). No spaces, underscores, digits.\n"
                    "2. Semantics: Keys MUST be high-level descriptive summaries of the action taken in that step, avoid question specific wording.\n"
                    "3. FORBIDDEN: Do NOT use generic sequential names (e.g., StepOne, Calculation1).\n\n"
                    "Rationale requirements:\n"
                    "1. StepReasoning field names must be human-readable\n"
                    "2. Each StepReasoning field must contain short, essential reasoning\n"
                )
                return prompt_text, True

            return system_prompt, user_prompt_builder

        if category_based and hasattr(prompt_module, "wrap_structured_category_based_freeform_prompt"):
            def user_prompt_builder(
                question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
            ) -> Tuple[str, bool]:
                answer_arg = gold_answer if include_gold_answer else None
                category = ""
                intents_text = ""
                if isinstance(sample_meta, dict):
                    category = str(sample_meta.get("group") or sample_meta.get("category") or "").strip()
                    intent_value = sample_meta.get("intent")
                    if isinstance(intent_value, list):
                        intents_text = ", ".join(str(item).strip() for item in intent_value if str(item).strip())
                    else:
                        intents_text = str(intent_value or "").strip()
                prompt_text = prompt_module.wrap_structured_category_based_freeform_prompt(
                    question=question,
                    category=category,
                    intents=intents_text,
                    answer=answer_arg,
                    task_type=task_type,
                    detailed=detailed,
                )
                return prompt_text.strip(), True

            return system_prompt, user_prompt_builder

        if not category_based and hasattr(prompt_module, "wrap_structured_freeform_prompt"):
            def user_prompt_builder(
                question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
            ) -> Tuple[str, bool]:
                answer_arg = gold_answer if include_gold_answer else None
                prompt_text = prompt_module.wrap_structured_freeform_prompt(
                    question=question,
                    answer=answer_arg,
                    task_type=task_type,
                    detailed=detailed,
                )
                return prompt_text.strip(), True
            return system_prompt, user_prompt_builder

    if normalized == "freeform" and hasattr(prompt_module, "wrap_freeform_prompt"):
        if task_type in text_task_types:
            system_prompt = ""
        def user_prompt_builder(
            question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
        ) -> Tuple[str, bool]:
            answer_arg = gold_answer if include_gold_answer else None
            prompt_text = prompt_module.wrap_freeform_prompt(
                question=question,
                answer=answer_arg,
                task_type=task_type,
                detailed=detailed,
            )
            return prompt_text.strip(), True
        return system_prompt, user_prompt_builder

    if normalized == "cot":
        cot_examples_block = part_three_prompt
        normal_base = str(getattr(prompt_module, "NORMAL_BASE_INSTRUCTIONS", "") or "").strip()

        def user_prompt_builder(
            question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
        ) -> Tuple[str, bool]:
            question_text = question.strip()
            prompt_sections = []
            if normal_base:
                prompt_sections.append(normal_base)
            if cot_examples_block:
                prompt_sections.append(cot_examples_block)
            prompt_sections.append(f"Question: {question_text}\nAnswer:")
            prompt_text = "\n\n".join(section for section in prompt_sections if section.strip())
            return prompt_text, True

        return system_prompt, user_prompt_builder

    prompt_parts = [part_two_prompt, part_three_prompt]
    if include_teacher_requirements:
        teacher_requirements = str(getattr(prompt_module, "TEACHER_REQUIREMENTS", "") or "").strip()
        if teacher_requirements:
            prompt_parts.append(teacher_requirements)

    combined_prompt_parts = [part.strip() for part in prompt_parts if part.strip()]
    if not combined_prompt_parts:
        raise SystemExit("Strategy prompt is empty; cannot build user instructions.")
    combined_prompt = "\n\n".join(combined_prompt_parts)

    def user_prompt_builder(
        question: str, gold_answer: str | None, _: str = "math", sample_meta: dict | None = None
    ) -> Tuple[str, bool]:
        if normalized == "normal":
            tail = (
                "\n\nLet's think step by step to solve the problem.\n\n"
                "Rules:\n- Finish the last answer with the numeric answer.\n"
                "- Keep reasoning concise and focused on the calculation."
            )
            return f"{combined_prompt}\n\nQ: {question}\nA:{tail}", True
        return combined_prompt, False

    return system_prompt, user_prompt_builder


def run_samples(
    args,
    strategy: str,
    system_prompt: str,
    user_prompt_builder: Callable[[str, str | None, str, dict | None], Tuple[str, bool]],
    ask_fn: Callable[[str, str], str],
    model_name: str,
    request_retries: int = 3,
    raw_output_path: Path | None = None,
) -> int:
    """Common sampling loop used by GPT and Llama scripts."""
    dataset_name = getattr(args, "dataset_name", None)
    samples = load_samples(args.dataset_path, args.sample_index, args.num_samples)
    strategy_label = sanitize_identifier(strategy, "strategy", 0)
    results_path = args.output_dir / f"results_{strategy_label}.jsonl"
    existing_entries, processed_ids = load_existing_results(results_path)
    all_entries: List[Dict[str, Any]] = list(existing_entries)
    entry_ids: List[str] = [_entry_key(entry) for entry in all_entries]
    new_entries_added = 0

    for offset, (absolute_index, raw_sample) in enumerate(samples, start=1):
        sample = _apply_dataset_normalization(raw_sample, dataset_name)
        question = sample.get("question", "").strip()
        options_text = _extract_options_text(sample)
        gold_answer_raw = sample.get("answer", "")
        gold_answer_text = str(gold_answer_raw).strip()
        gold_rationale, gold_answer = split_answer_and_rationale(gold_answer_text)
        sample_id_value = sample.get("id")
        if not sample_id_value:
            sample_id_value = sample_identifier(sample, args.dataset_path, absolute_index)
        sample_id = str(sample_id_value)

        if sample_id in processed_ids:
            print(
                f"Sample {offset}/{args.num_samples}  |  Row #{absolute_index}  |  ID: {sample_id} already processed; skipping."
            )
            continue

        category_based = getattr(args, "category_based", False)
        ds_path_text = str(getattr(args, "dataset_path", "")).lower()
        is_gsm8k_dataset = ("gsm8k" in ds_path_text) or (dataset_name and str(dataset_name).lower() == "gsm8k")
        category_value = ""
        group_value = ""
        intent_value = ""
        intent_list_for_route: List[str] = []
        if category_based:
            category_intent_block = sample.get("category_intent", {})
            category_value = str(
                sample.get("group")
                or sample.get("category")
                or (category_intent_block.get("category") if isinstance(category_intent_block, dict) else "")
                or ""
            ).strip()

            intent_raw = sample.get("intent")
            if intent_raw is None and isinstance(category_intent_block, dict):
                intent_raw = category_intent_block.get("intent")
            if isinstance(intent_raw, list):
                intent_list_for_route = [str(item).strip() for item in intent_raw if str(item).strip()]
                intent_value = ", ".join(intent_list_for_route)
            else:
                intent_str = str(intent_raw or "").strip()
                if intent_str:
                    intent_list_for_route = [intent_str]
                    intent_value = intent_str
            group_value = str(sample.get("group") or "").strip() if is_gsm8k_dataset else ""

        task_instruction = sample.get("_task_instruction", "")
        use_text_role = args.task_type in {"common", "commonsense", "text"}
        effective_system_prompt = system_prompt

        if task_instruction and use_text_role:
            effective_system_prompt = task_instruction
            question_for_prompt = question
        else:
            question_for_prompt = f"{task_instruction}\n\n{question}" if task_instruction else question
        question_for_prompt = _append_options(question_for_prompt, options_text)
        answer_schema = _infer_choice_answer_schema(
            sample=sample,
            options_text=options_text,
            dataset_name=dataset_name,
            dataset_path=getattr(args, "dataset_path", None),
        )
        if strategy == "sgft":
            answer_schema = None
        if answer_schema:
            question_for_prompt = _append_choice_instruction(question_for_prompt, answer_schema)

        question_display_lines = [question_for_prompt]
        if category_based:
            if category_value:
                question_display_lines.append(f"Category: {category_value}")
            if intent_value:
                question_display_lines.append(f"Intents: {intent_value}")
        question_display = "\n".join(line for line in question_display_lines if line.strip())

        print(f"Sample {offset}/{args.num_samples}  |  Row #{absolute_index}  |  ID: {sample_id}")
        print(f"Question:\n{question_display}\n")
        if gold_answer_text:
            print("Ground-truth answer:")
            print(gold_answer_text)
            print()

        user_prompt_text, includes_question = user_prompt_builder(
            question_for_prompt, gold_answer, args.task_type, sample
        )
        if answer_schema:
            user_prompt_text = _apply_answer_schema(user_prompt_text, answer_schema)
        if includes_question:
            use_default = bool(effective_system_prompt and effective_system_prompt.strip())
            formatted_system_prompt = ""
            if use_default:
                formatted_system_prompt = f"\n{effective_system_prompt.strip() or 'You are a helpful math tutor.'}"
            formatted_user_prompt = user_prompt_text
        else:
            use_default = bool(effective_system_prompt and effective_system_prompt.strip())
            formatted_system_prompt, formatted_user_prompt = build_prompt_texts(
                effective_system_prompt, user_prompt_text, question_display, use_system_default=use_default
            )

        print("---- Prompt Start ----")
        print(formatted_system_prompt)
        print()
        print(formatted_user_prompt)
        print("---- Prompt End ----")
        print()

        if args.show_only:
            continue

        max_attempts = request_retries
        model_reply = None
        for attempt in range(1, max_attempts + 1):
            try:
                model_reply = ask_fn(formatted_system_prompt, formatted_user_prompt)
                print("Model response:")
                print(model_reply)
                print()
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                print(
                    f"Request failed for sample {sample_id} "
                    f"(attempt {attempt}/{max_attempts}): {exc.__class__.__name__}: {exc}"
                )
                if attempt < max_attempts:
                    sleep_seconds = min(2 * attempt, 10)
                    print(f"Retrying in {sleep_seconds} seconds...\n")
                    time.sleep(sleep_seconds)
                else:
                    print("Max attempts reached; moving to next sample.\n")

        if model_reply is None:
            print(
                f"Failed to obtain a response for sample {sample_id}; recording the error and continuing."
            )
            continue

        if raw_output_path is not None:
            append_raw_response(raw_output_path, model_reply)

        rationale, parsed_ans, parsed_payload = parse_model_output(model_reply)
        if answer_schema:
            mapped_choice = _map_answer_to_choice(parsed_ans, options_text)
            if mapped_choice is not None:
                parsed_ans = mapped_choice
            if not _is_choice_letter(parsed_ans):
                choice_from_rationale = _extract_choice_from_text(rationale)
                if choice_from_rationale:
                    parsed_ans = choice_from_rationale
            if not _is_choice_letter(parsed_ans):
                choice_from_reply = _extract_choice_from_text(model_reply)
                if choice_from_reply:
                    parsed_ans = choice_from_reply
            if isinstance(parsed_payload, dict) and parsed_ans is not None:
                parsed_payload["ans"] = parsed_ans
        rationale_value = model_reply if rationale is None else rationale
        response_payload = parsed_payload if parsed_payload is not None else model_reply

        augmented_route = None
        if isinstance(parsed_payload, dict):
            route_payload = parsed_payload.get("route")
            if isinstance(route_payload, dict):
                augmented_route = dict(route_payload)
            elif category_based:
                augmented_route = {}
            if category_based and augmented_route is not None:
                if category_value:
                    augmented_route.setdefault("category", category_value)
                if intent_list_for_route:
                    augmented_route.setdefault("intent", intent_list_for_route)
                if is_gsm8k_dataset and group_value:
                    augmented_route.setdefault("group", group_value)
                parsed_payload["route"] = augmented_route

        if augmented_route is not None:
            response_payload = parsed_payload

        route_for_record = None
        if isinstance(response_payload, dict):
            route_for_record = response_payload.get("route")
        if route_for_record is None and category_based:
            route_for_record = {
                "category": category_value,
                "intent": intent_list_for_route,
            }
            if is_gsm8k_dataset and group_value:
                route_for_record["group"] = group_value
        elif category_based and isinstance(route_for_record, dict):
            if category_value:
                route_for_record.setdefault("category", category_value)
            if intent_list_for_route:
                route_for_record.setdefault("intent", intent_list_for_route)
            if is_gsm8k_dataset and group_value:
                route_for_record.setdefault("group", group_value)

        matches_gold = False
        if gold_answer is not None:
            matches_gold = _answers_match(gold_answer, parsed_ans)
            if not matches_gold and isinstance(response_payload, str):
                matches_gold = gold_answer.strip().lower() in response_payload.lower()

        sample_record = {
            "index": sample_id,
            "question": question,
            "answer_from_dataset": gold_answer_text,
            "gold_ans": gold_answer,
            "gold": gold_answer,
            "prompt_strategy": strategy,
            "model": model_name,
            "route": route_for_record,
            "response_payload": response_payload,
            "response_rationale": rationale_value,
            "response_ans": parsed_ans,
            "gold_matches_answer": matches_gold,
        }
        if options_text:
            sample_record["options"] = options_text
        insert_at = bisect_left(entry_ids, sample_id)
        entry_ids.insert(insert_at, sample_id)
        all_entries.insert(insert_at, sample_record)
        write_results(results_path, all_entries)
        processed_ids.add(sample_id)
        new_entries_added += 1
        print(f"Saved progress to {results_path}\n")

    if args.show_only:
        print("--show-only flag set; no API requests were made and no files were written.")
        return 0

    if new_entries_added:
        print(f"Processing complete. Wrote {new_entries_added} new entries to {results_path}")
    else:
        if results_path.exists():
            print(f"No new samples processed; existing entries remain in {results_path}")
        else:
            print("No samples were processed; results file was not created.")
    return new_entries_added


def append_raw_response(path: Path, response_text: str) -> None:
    """Append a raw model response to a text file, separated by blank lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    has_existing = path.exists() and path.stat().st_size > 0
    payload = response_text.rstrip()
    with path.open("a", encoding="utf-8") as handle:
        if has_existing:
            handle.write("\n\n")
        handle.write(payload)
        handle.write("\n")


def parse_model_output(response_text: str) -> Tuple[Any, Any, Dict[str, Any] | None]:
    """Extract rationale, answer, and full payload from a model's JSON output.

    Attempts to repair slightly invalid JSON (e.g., single quotes) using
    ast.literal_eval if json.loads fails. Falls back to extracting an answer
    from XML <Answer> tags. Returns (None, None, None) when parsing cannot be
    recovered so callers can keep raw text and leave response_ans null.
    """
    if not isinstance(response_text, str):
        return None, None, None

    json_text = _extract_json_text(response_text)
    if json_text is None:
        xml_answer = _extract_xml_answer_value(response_text)
        if xml_answer is not None:
            return None, xml_answer, None
        trailing_num = _extract_trailing_number(response_text)
        if trailing_num is not None:
            return response_text, trailing_num, None
        return None, None, None

    data = _parse_json_or_literal(json_text)
    if data is None:
        xml_answer = _extract_xml_answer_value(response_text)
        if xml_answer is not None:
            return None, xml_answer, None
        trailing_num = _extract_trailing_number(response_text)
        if trailing_num is not None:
            return response_text, trailing_num, None
        return None, None, None

    rationale = data.get("rationale")
    answer = _normalize_answer_token(data.get("ans"))
    return rationale, answer, data


def _extract_json_text(response_text: str) -> str | None:
    """Pull a JSON-like block from the response text."""
    match = re.search(r"```json\n({.*?})\n```", response_text, re.DOTALL)
    if match:
        return match.group(1)

    brace_start = response_text.find("{")
    brace_end = response_text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_start < brace_end:
        return response_text[brace_start : brace_end + 1]
    return None


def _extract_xml_answer_value(response_text: str) -> Any:
    """Extract an answer token from tags or common answer markers."""
    if not isinstance(response_text, str):
        return None
    match = ANSWER_TAG_RE.search(response_text)
    if match:
        return _normalize_answer_token(match.group(1))

    ans_match = ANS_FIELD_RE.search(response_text)
    if ans_match:
        token = ans_match.group(2) or ans_match.group(3) or ans_match.group(4) or ""
        value = _normalize_answer_token(token)
        if value is not None:
            return value

    for pattern in ANSWER_CHOICE_PATTERNS:
        choice_match = pattern.search(response_text)
        if choice_match:
            return choice_match.group(1).upper()
    option_match = OPTION_CHOICE_RE.search(response_text)
    if option_match:
        return option_match.group(1).upper()
    return None


def _extract_trailing_number(response_text: str) -> Any:
    """Extract the last numeric token from free-form text (e.g., 'Answer:72')."""
    if not isinstance(response_text, str):
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", response_text.replace(",", ""))
    if not matches:
        return None
    return _coerce_number(matches[-1])


def _parse_json_or_literal(json_text: str) -> Dict[str, Any] | None:
    """Try strict JSON first, then fall back to literal_eval for common issues."""
    try:
        return json.loads(json_text)
    except (TypeError, json.JSONDecodeError):
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            candidate = ast.literal_eval(json_text)
        if isinstance(candidate, dict):
            return candidate
    except Exception:
        pass
    return None

def _content_chunks(response_obj) -> List[str]:
    """Yield text chunks from different model response shapes."""
    chunks: List[str] = []
    output_text = getattr(response_obj, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        chunks.append(output_text.strip())
        return chunks

    choices = getattr(response_obj, "choices", None)
    if choices:
        for choice in choices:
            message = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message") or choice.get("delta")

            content = None
            if message is not None:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
            if content is None:
                content = getattr(choice, "content", None)
                if content is None and isinstance(choice, dict):
                    content = choice.get("content")

            if isinstance(content, list):
                for chunk in content:
                    chunk_type = getattr(chunk, "type", None) or chunk.get("type")
                    if chunk_type and "text" not in chunk_type and chunk_type != "output_text":
                        continue
                    text_value = getattr(chunk, "text", None) or chunk.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
            elif isinstance(content, str):
                chunks.append(content)
        if chunks:
            return chunks

    output = getattr(response_obj, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None) or getattr(item, "text", None)
            if isinstance(content, list):
                for chunk in content:
                    chunk_type = getattr(chunk, "type", None) or chunk.get("type")
                    if chunk_type and "text" not in chunk_type and chunk_type != "output_text":
                        continue
                    text_value = getattr(chunk, "text", None) or chunk.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
            elif isinstance(content, str):
                chunks.append(content)
        return chunks

    content = getattr(response_obj, "content", None)
    if isinstance(content, list):
        for chunk in content:
            text_value = getattr(chunk, "text", None) or chunk.get("text")
            if isinstance(text_value, str):
                chunks.append(text_value)
    return chunks


def extract_response_text(response) -> str:
    """Convert a model response payload into plain text."""
    chunks = _content_chunks(response)
    if chunks:
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return str(response)


def sanitize_identifier(value: str, dataset_name: str, absolute_index: int) -> str:
    """Create a filesystem-safe identifier for saving responses."""
    candidate = SANITIZE_RE.sub("_", value).strip("._")
    if candidate:
        return candidate
    return f"{dataset_name}_{absolute_index:05d}"


def sample_identifier(sample: dict, dataset_path: Path, absolute_index: int) -> str:
    """Determine a stable identifier for a dataset row."""
    dataset_name = dataset_path.stem or "sample"
    for key in ID_FIELDS:
        if key in sample and sample[key] not in (None, ""):
            return sanitize_identifier(str(sample[key]), dataset_name, absolute_index)
    return f"{dataset_name}_{absolute_index:05d}"

def split_answer_and_rationale(text: str) -> tuple[str, str]:
    """Split a string into a rationale and a final answer."""
    if not isinstance(text, str):
        return "", ""
    parts = text.split("####")
    if len(parts) == 1:
        return "", parts[0].strip()
    rationale = parts[0].strip()
    answer = parts[1].strip()
    return rationale, answer
