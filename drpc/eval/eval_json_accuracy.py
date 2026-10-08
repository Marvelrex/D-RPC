#!/usr/bin/env python3
"""
Compute answer accuracy for a JSON/JSONL predictions file or all JSON/JSONL
files in a directory (SVAMP/GSM8K/AQUA/StrategyQA etc).

Each row is expected to include:
- gold_ans: reference answer (fallbacks: gold_answer_extracted, gold, answer_from_dataset, answer, gold_answer, gold_rationale)
- model_response: model output, either a dict or a JSON string containing an "ans" field
- ans/response_ans: direct prediction fields (numeric, bool, or choice letter)

The script tolerates JSON arrays or JSONL input. Answers are normalized:
- Numeric with tolerance when possible
- Booleans normalized to true/false
- Choice letters (A/B/C/...) compared case-insensitively
- Otherwise lowercase string comparison
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

CHOICE_PATTERN = re.compile(r"^[A-Ea-e]$")
CHOICE_TOKEN_PATTERN = re.compile(r"\b([A-E])\b", re.IGNORECASE)
OPTION_LINE_PATTERN = re.compile(
    r"(?:^|\n)\s*([A-E])\s*[\)\.\:\-]\s*([^\n\r]+)", re.IGNORECASE
)
ANSWER_CHOICE_PATTERNS = (
    re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE),
    re.compile(r"\banswer\b(?:\s+|[:=\-]\s*)([A-E])\b", re.IGNORECASE),
    re.compile(r"\boption\b(?:\s+|[:=\-]\s*)([A-E])\b", re.IGNORECASE),
)
ANSWER_CHOICE_NUMERIC_PATTERNS = (
    re.compile(r"\banswer\b(?:\s+|[:=\-]\s*)([1-5])(?:\.0+)?\b", re.IGNORECASE),
    re.compile(r"\boption\b(?:\s+|[:=\-]\s*)([1-5])(?:\.0+)?\b", re.IGNORECASE),
    re.compile(r"\bchoice\b(?:\s+|[:=\-]\s*)([1-5])(?:\.0+)?\b", re.IGNORECASE),
)
ANSWER_TAG_PATTERN = re.compile(
    r"(?:<|\u27e8)\s*answer\s*(?:>|\u27e9)(.*?)(?:<|\u27e8)\s*/\s*answer\s*(?:>|\u27e9)",
    re.IGNORECASE | re.DOTALL,
)
ANSWER_TEXT_PATTERN = re.compile(
    r"(?:final answer|the final answer|answer is|answer:)\s*([^\n\r]+)",
    re.IGNORECASE,
)
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
BOOL_TOKEN_PATTERN = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
ANS_KEY_PATTERN = re.compile(r'(?i)(?:^|[\s{,])(?:\"ans\"|ans)\s*:\s*')
NUMERIC_CHOICE_PATTERN = re.compile(r"^[1-5](?:\.0+)?$")
FRACTION_TEXT_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$")
LATEX_FRAC_BRACED_PATTERN = re.compile(r"^\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}$")
LATEX_FRAC_COMPACT_PATTERN = re.compile(r"^\\frac([+-]?\d+(?:\.\d+)?)([+-]?\d+(?:\.\d+)?)$")


def load_rows(path: Path) -> List[dict]:
    """Load rows from JSON array or JSONL, skipping malformed lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Could not read file: {exc}") from exc

    if not text.strip():
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    rows: List[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception as exc:
            print(f"Skipping line {lineno}: {exc}", file=sys.stderr)
    return rows


def _coerce_number(value: object) -> Optional[float]:
    """Attempt to coerce a value to float, returning None on failure."""
    def _parse_float_token(token: str) -> Optional[float]:
        s = token.strip().replace(",", "").replace("−", "-")
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _parse_fraction(s: str) -> Optional[float]:
        candidate = s.strip().replace("−", "-")
        if not candidate:
            return None
        sign = 1.0
        if candidate[0] in "+-":
            sign = -1.0 if candidate[0] == "-" else 1.0
            candidate = candidate[1:].strip()
            if not candidate:
                return None

        m = LATEX_FRAC_BRACED_PATTERN.fullmatch(candidate)
        if m:
            num = _parse_float_token(m.group(1))
            den = _parse_float_token(m.group(2))
            if num is None or den in (None, 0.0):
                return None
            out = sign * (num / den)
            return out if math.isfinite(out) else None

        compact = candidate.replace(" ", "")
        m = LATEX_FRAC_COMPACT_PATTERN.fullmatch(compact)
        if m:
            num = _parse_float_token(m.group(1))
            den = _parse_float_token(m.group(2))
            if num is None or den in (None, 0.0):
                return None
            out = sign * (num / den)
            return out if math.isfinite(out) else None

        m = FRACTION_TEXT_PATTERN.fullmatch(candidate)
        if m:
            num = _parse_float_token(m.group(1))
            den = _parse_float_token(m.group(2))
            if num is None or den in (None, 0.0):
                return None
            out = sign * (num / den)
            return out if math.isfinite(out) else None
        return None

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:
            return None
    if isinstance(value, str):
        s = value.strip()
        if "####" in s:
            s = s.split("####")[-1].strip()
        s = s.rstrip(".,;:")
        frac = _parse_fraction(s)
        if frac is not None:
            return frac
        try:
            return float(s)
        except Exception:
            match = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
            if match:
                try:
                    return float(match.group())
                except Exception:
                    return None
    return None


def normalize_answer(value: object) -> object:
    """Return normalized answer: number, bool, choice letter, else trimmed lower string."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"yes", "no"}:
            return lowered == "yes"
        if lowered in {"true", "false"}:
            return lowered == "true"
        if CHOICE_PATTERN.match(stripped):
            return stripped.upper()
    num = _coerce_number(value)
    if num is not None:
        return num
    return str(value).strip().lower()


def _labelize(value: object) -> str:
    """Convert a normalized answer into a stable string label."""
    if value is None:
        return "__MISSING__"
    normalized = normalize_answer(value)
    if normalized is None:
        return "__MISSING__"
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, (int, float)):
        return f"{float(normalized):.12g}"
    return str(normalized)


def _extract_choice_from_text(text: str) -> Optional[str]:
    """Extract an A-E choice letter from free-form text."""
    if not isinstance(text, str):
        return None
    for pattern in ANSWER_CHOICE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    for pattern in ANSWER_CHOICE_NUMERIC_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        mapped = _choice_from_numeric(match.group(1))
        if mapped is not None:
            return mapped
    return None


def _extract_answer_from_text(text: str) -> Optional[str]:
    """Extract answer content between <Answer>...</Answer> or ⟨Answer⟩...⟨/Answer⟩ tags."""
    if not isinstance(text, str):
        return None
    match = ANSWER_TAG_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _coerce_bool_text(text: str) -> Optional[bool]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip().strip("\"'")
    if not cleaned:
        return None
    cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cleaned)
    lowered = cleaned.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    return None


def _coerce_choice_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip().strip("\"'")
    if not cleaned:
        return None
    match = CHOICE_TOKEN_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    return None


def _choice_from_numeric(value: object) -> Optional[str]:
    choice_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return choice_map.get(str(value))
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return choice_map.get(str(int(value)))
    if isinstance(value, str):
        cleaned = value.strip().strip("\"'")
        if NUMERIC_CHOICE_PATTERN.match(cleaned):
            return choice_map.get(cleaned.split(".", 1)[0])
    return None


def _extract_bool_from_text(text: str) -> Optional[bool]:
    if not isinstance(text, str):
        return None
    answer_match = ANSWER_TEXT_PATTERN.search(text)
    if not answer_match:
        return None
    return _coerce_bool_text(answer_match.group(1))


def _extract_last_number(text: str) -> Optional[str]:
    """Return the last numeric token in text, if any."""
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1]


def _extract_final_numeric(text: str) -> Optional[str]:
    """Extract a final numeric answer from text, preferring explicit answer cues."""
    if not isinstance(text, str):
        return None
    if not NUMBER_PATTERN.search(text):
        return None

    boxed = BOXED_PATTERN.search(text)
    if boxed:
        boxed_val = _extract_last_number(boxed.group(1))
        if boxed_val is not None:
            return boxed_val

    answer_match = ANSWER_TEXT_PATTERN.search(text)
    if answer_match:
        answer_text = answer_match.group(1)
        answer_val = _extract_last_number(answer_text)
        if answer_val is not None:
            return answer_val

    if BOOL_TOKEN_PATTERN.search(text):
        return None

    return _extract_last_number(text)


def _extract_ans_field(text: str) -> Optional[str]:
    """Extract ans field from JSON-ish text, tolerating escaped quotes."""
    if not isinstance(text, str):
        return None
    attempts = [text]
    unescaped = text.replace('\\"', '"')
    if unescaped != text:
        attempts.append(unescaped)

    for candidate in attempts:
        for match in ANS_KEY_PATTERN.finditer(candidate):
            rest = candidate[match.end() :].lstrip()
            if not rest:
                continue
            if rest[0] in {"'", '"'}:
                quote = rest[0]
                value_chars: List[str] = []
                escaped = False
                for ch in rest[1:]:
                    if escaped:
                        value_chars.append(ch)
                        escaped = False
                        continue
                    if ch == "\\":
                        escaped = True
                        continue
                    if ch == quote:
                        break
                    value_chars.append(ch)
                value = "".join(value_chars).strip()
                if value:
                    return value
                continue

            value_chars = []
            for ch in rest:
                if ch in {",", "}", "\n", "\r"}:
                    break
                value_chars.append(ch)
            value = "".join(value_chars).strip().strip('"').strip("'")
            if value:
                return value
    return None


def _extract_embedded_json_ans(text: str) -> object:
    """Extract ans from a JSON object embedded in text (e.g., after <think> blocks)."""
    if not isinstance(text, str):
        return None

    def _ans_from_obj(obj: object) -> object:
        if isinstance(obj, dict) and "ans" in obj:
            return obj.get("ans")
        return None

    attempts = [text]
    if '\\"' in text:
        attempts.append(text.replace('\\"', '"'))

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if parsed is not None:
            ans = _ans_from_obj(parsed)
            if ans is not None:
                return ans
            if isinstance(parsed, str):
                try:
                    inner = json.loads(parsed)
                except Exception:
                    inner = None
                ans = _ans_from_obj(inner)
                if ans is not None:
                    return ans

        brace_start = candidate.find("{")
        brace_end = candidate.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_start < brace_end:
            snippet = candidate[brace_start : brace_end + 1]
            try:
                parsed_snippet = json.loads(snippet)
            except Exception:
                parsed_snippet = None
            ans = _ans_from_obj(parsed_snippet)
            if ans is not None:
                return ans
    return None


def _extract_choice_from_row(row: dict) -> Optional[str]:
    """Best-effort choice extraction from model output fields."""
    for key in ("model_response", "response_payload", "response_rationale"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            for text_key in ("rationale", "response_rationale", "content"):
                text_val = val.get(text_key)
                if isinstance(text_val, str):
                    choice = _extract_choice_from_text(text_val)
                    if choice:
                        return choice
            choice = _extract_choice_from_text(json.dumps(val, ensure_ascii=False))
            if choice:
                return choice
        elif isinstance(val, str):
            choice = _extract_choice_from_text(val)
            if choice:
                return choice
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    text_val = parsed.get("rationale")
                    if isinstance(text_val, str):
                        choice = _extract_choice_from_text(text_val)
                        if choice:
                            return choice
            except Exception:
                pass
    return None


def _extract_option_map(row: dict) -> dict[str, str]:
    """Parse A-E option text from common options fields on a row."""
    for key in ("options", "Options", "option", "Option"):
        val = row.get(key)
        if not isinstance(val, str):
            continue
        matches = OPTION_LINE_PATTERN.findall(val)
        if not matches:
            continue
        out: dict[str, str] = {}
        for letter, option_text in matches:
            out[letter.upper()] = option_text.strip()
        if out:
            return out
    return {}


def _map_pred_to_option_choice(row: dict, pred: object) -> Optional[str]:
    """Map a non-choice prediction to an option letter via option values."""
    option_map = _extract_option_map(row)
    if not option_map:
        return None

    matches: List[str] = []
    for letter, option_text in option_map.items():
        if answers_match(option_text, pred):
            matches.append(letter)
    if len(matches) == 1:
        return matches[0]
    return None


def _looks_like_choice_task(row: dict) -> bool:
    gold = extract_gold_ans(row)
    if isinstance(gold, str) and CHOICE_PATTERN.match(gold.strip()):
        return True
    return bool(_extract_option_map(row))


def extract_gold_ans(row: dict) -> object:
    for key in (
        "gold_ans",
        "gold_answer_extracted",
        "gold",
        "answer_from_dataset",
        "answer",
        "gold_answer",
    ):
        if key in row:
            val = row[key]
            if isinstance(val, str) and not val.strip():
                continue
            return val
    if "gold_rationale" in row:
        rat = row["gold_rationale"]
        if isinstance(rat, str) and not rat.strip():
            return None
        num = _coerce_number(rat)
        return num if num is not None else rat
    return None


def extract_pred_ans(row: dict) -> object:
    choice_task = _looks_like_choice_task(row)

    def _maybe_choice_map(value: object) -> object:
        if not choice_task:
            return value
        mapped = _choice_from_numeric(value)
        if mapped is not None:
            return mapped
        if isinstance(value, str):
            text_choice = _extract_choice_from_text(value)
            if text_choice is not None:
                return text_choice
            direct_choice = _coerce_choice_text(value)
            if direct_choice is not None:
                return direct_choice
        return value

    for key in ("model_ans", "pred_ans", "response_ans"):
        if key in row and row[key] is not None:
            return _maybe_choice_map(row[key])
    resp_payload = row.get("response_payload")
    if isinstance(resp_payload, dict):
        maybe = resp_payload.get("ans")
        if maybe is not None:
            return _maybe_choice_map(maybe)
    if isinstance(resp_payload, str):
        json_ans = _extract_embedded_json_ans(resp_payload)
        if json_ans is None:
            json_ans = _extract_ans_field(resp_payload)
        if json_ans is not None:
            return _maybe_choice_map(json_ans)
        tag_ans = _extract_answer_from_text(resp_payload)
        if tag_ans is not None:
            choice_ans = _coerce_choice_text(tag_ans)
            if choice_ans is not None:
                return choice_ans
            if choice_task:
                numeric_choice = _choice_from_numeric(tag_ans)
                if numeric_choice is not None:
                    return numeric_choice
            bool_ans = _coerce_bool_text(tag_ans)
            if bool_ans is not None:
                return bool_ans
            return tag_ans
        bool_ans = _extract_bool_from_text(resp_payload)
        if bool_ans is not None:
            return bool_ans
        num_ans = _extract_final_numeric(resp_payload)
        if num_ans is not None:
            if choice_task:
                numeric_choice = _choice_from_numeric(num_ans)
                if numeric_choice is not None:
                    return numeric_choice
            else:
                return num_ans
        coerced = _coerce_number(resp_payload)
        if coerced is not None:
            if choice_task:
                numeric_choice = _choice_from_numeric(coerced)
                if numeric_choice is not None:
                    return numeric_choice
            else:
                return coerced
    resp = row.get("model_response")
    if isinstance(resp, dict):
        direct_ans = resp.get("ans")
        if direct_ans is not None:
            return _maybe_choice_map(direct_ans)
        for text_key in ("rationale", "response_rationale", "content"):
            text_val = resp.get(text_key)
            if isinstance(text_val, str):
                tag_ans = _extract_answer_from_text(text_val)
                if tag_ans is not None:
                    choice_ans = _coerce_choice_text(tag_ans)
                    if choice_ans is not None:
                        return choice_ans
                    if choice_task:
                        numeric_choice = _choice_from_numeric(tag_ans)
                        if numeric_choice is not None:
                            return numeric_choice
                    bool_ans = _coerce_bool_text(tag_ans)
                    if bool_ans is not None:
                        return bool_ans
                    return tag_ans
                bool_ans = _extract_bool_from_text(text_val)
                if bool_ans is not None:
                    return bool_ans
                num_ans = _extract_final_numeric(text_val)
                if num_ans is not None:
                    if choice_task:
                        numeric_choice = _choice_from_numeric(num_ans)
                        if numeric_choice is not None:
                            return numeric_choice
                    else:
                        return num_ans
        resp_text = json.dumps(resp, ensure_ascii=False)
        tag_ans = _extract_answer_from_text(resp_text)
        if tag_ans is not None:
            choice_ans = _coerce_choice_text(tag_ans)
            if choice_ans is not None:
                return choice_ans
            if choice_task:
                numeric_choice = _choice_from_numeric(tag_ans)
                if numeric_choice is not None:
                    return numeric_choice
            bool_ans = _coerce_bool_text(tag_ans)
            if bool_ans is not None:
                return bool_ans
            return tag_ans
        bool_ans = _extract_bool_from_text(resp_text)
        if bool_ans is not None:
            return bool_ans
        num_ans = _extract_final_numeric(resp_text)
        if num_ans is not None:
            if choice_task:
                numeric_choice = _choice_from_numeric(num_ans)
                if numeric_choice is not None:
                    return numeric_choice
            else:
                return num_ans
        return None
    if isinstance(resp, str):
        json_ans = _extract_embedded_json_ans(resp)
        if json_ans is None:
            json_ans = _extract_ans_field(resp)
        if json_ans is not None:
            return _maybe_choice_map(json_ans)
        tag_ans = _extract_answer_from_text(resp)
        if tag_ans is not None:
            choice_ans = _coerce_choice_text(tag_ans)
            if choice_ans is not None:
                return choice_ans
            if choice_task:
                numeric_choice = _choice_from_numeric(tag_ans)
                if numeric_choice is not None:
                    return numeric_choice
            bool_ans = _coerce_bool_text(tag_ans)
            if bool_ans is not None:
                return bool_ans
            return tag_ans
        bool_ans = _extract_bool_from_text(resp)
        if bool_ans is not None:
            return bool_ans
        num_ans = _extract_final_numeric(resp)
        if num_ans is not None:
            if choice_task:
                numeric_choice = _choice_from_numeric(num_ans)
                if numeric_choice is not None:
                    return numeric_choice
            else:
                return num_ans
    has_model_output = any(k in row for k in ("model_response", "model_ans", "response_payload"))
    if has_model_output:
        return None
    for key in ("ans", "response_ans", "pred_ans"):
        if key in row and row[key] is not None:
            return _maybe_choice_map(row[key])
    resp_rat = row.get("response_rationale")
    if isinstance(resp_rat, str):
        tag_ans = _extract_answer_from_text(resp_rat)
        if tag_ans is not None:
            choice_ans = _coerce_choice_text(tag_ans)
            if choice_ans is not None:
                return choice_ans
            if choice_task:
                numeric_choice = _choice_from_numeric(tag_ans)
                if numeric_choice is not None:
                    return numeric_choice
            bool_ans = _coerce_bool_text(tag_ans)
            if bool_ans is not None:
                return bool_ans
            return tag_ans
        bool_ans = _extract_bool_from_text(resp_rat)
        if bool_ans is not None:
            return bool_ans
        num_ans = _extract_final_numeric(resp_rat)
        if num_ans is not None:
            if choice_task:
                numeric_choice = _choice_from_numeric(num_ans)
                if numeric_choice is not None:
                    return numeric_choice
            else:
                return num_ans
        num = _coerce_number(resp_rat)
        if num is not None:
            if choice_task:
                numeric_choice = _choice_from_numeric(num)
                if numeric_choice is not None:
                    return numeric_choice
            else:
                return num
    return None


def answers_match(gold: object, pred: object, tol: float = 1e-6) -> bool:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)

    if isinstance(gold_norm, bool) and isinstance(pred_norm, bool):
        return gold_norm is pred_norm

    if isinstance(gold_norm, str) and CHOICE_PATTERN.match(gold_norm):
        return isinstance(pred_norm, str) and gold_norm == pred_norm.upper()

    if isinstance(gold_norm, (int, float)) and isinstance(pred_norm, (int, float)):
        return math.isclose(float(gold_norm), float(pred_norm), rel_tol=tol, abs_tol=tol)

    return gold_norm == pred_norm


def extract_json_ans(row: dict) -> object:
    """Extract ans from JSON objects in response_payload or model_response."""
    for key in ("response_payload", "model_response"):
        val = row.get(key)
        if isinstance(val, dict):
            if "ans" in val:
                return val["ans"]
            continue
        if isinstance(val, str):
            embedded = _extract_embedded_json_ans(val)
            if embedded is not None:
                return embedded
            field_ans = _extract_ans_field(val)
            if field_ans is not None:
                return field_ans
    return None


def compute_format_validity(rows: Iterable[dict]) -> tuple[int, int, float]:
    """Return (parsed_correct, parsed_total, ratio) for JSON-parsed answers."""
    parsed_total = 0
    parsed_correct = 0
    for row in rows:
        parsed_ans = extract_json_ans(row)
        if parsed_ans is None:
            continue
        gold = extract_gold_ans(row)
        if gold is None:
            continue
        parsed_total += 1
        if answers_match(gold, parsed_ans):
            parsed_correct += 1
    ratio = parsed_correct / parsed_total if parsed_total else 0.0
    return parsed_correct, parsed_total, ratio


def compute_parse_stats(rows: Iterable[dict]) -> tuple[int, int, float]:
    """Return (parsed_total, total, ratio) for JSON-parsed answers."""
    parsed_total = 0
    total = 0
    for row in rows:
        total += 1
        if extract_json_ans(row) is not None:
            parsed_total += 1
    ratio = parsed_total / total if total else 0.0
    return parsed_total, total, ratio


def evaluate(rows: Iterable[dict]) -> tuple[int, int, List[Tuple[int, object, object]]]:
    rows_list = list(rows)
    total = len(rows_list)
    correct = 0
    misses: List[Tuple[int, object, object]] = []
    for idx, row in enumerate(rows_list):
        gold = extract_gold_ans(row)
        pred = extract_pred_ans(row)
        if isinstance(gold, str) and CHOICE_PATTERN.match(gold.strip()):
            numeric_choice = _choice_from_numeric(pred)
            if numeric_choice is not None:
                pred = numeric_choice
            if isinstance(pred, str) and not CHOICE_PATTERN.match(pred.strip()):
                pred_text_choice = _extract_choice_from_text(pred)
                if pred_text_choice is not None:
                    pred = pred_text_choice
            if pred is not None and not (isinstance(pred, str) and CHOICE_PATTERN.match(pred.strip())):
                mapped_choice = _map_pred_to_option_choice(row, pred)
                if mapped_choice:
                    pred = mapped_choice
            if pred is None or not (isinstance(pred, str) and CHOICE_PATTERN.match(pred.strip())):
                fallback_choice = _extract_choice_from_row(row)
                if fallback_choice:
                    pred = fallback_choice
        if gold is None or pred is None:
            misses.append((idx, gold, pred))
            continue
        if answers_match(gold, pred):
            correct += 1
        else:
            misses.append((idx, gold, pred))
    return correct, total, misses


def evaluate_with_labels(
    rows: Iterable[dict],
) -> tuple[int, int, List[Tuple[int, object, object]], List[Tuple[str, str, bool]]]:
    """Evaluate rows and return (correct, total, misses, labels)."""
    rows_list = list(rows)
    total = len(rows_list)
    correct = 0
    misses: List[Tuple[int, object, object]] = []
    labels: List[Tuple[str, str, bool]] = []

    for idx, row in enumerate(rows_list):
        gold = extract_gold_ans(row)
        pred = extract_pred_ans(row)
        if isinstance(gold, str) and CHOICE_PATTERN.match(gold.strip()):
            numeric_choice = _choice_from_numeric(pred)
            if numeric_choice is not None:
                pred = numeric_choice
            if isinstance(pred, str) and not CHOICE_PATTERN.match(pred.strip()):
                pred_text_choice = _extract_choice_from_text(pred)
                if pred_text_choice is not None:
                    pred = pred_text_choice
            if pred is not None and not (isinstance(pred, str) and CHOICE_PATTERN.match(pred.strip())):
                mapped_choice = _map_pred_to_option_choice(row, pred)
                if mapped_choice:
                    pred = mapped_choice
            if pred is None or not (isinstance(pred, str) and CHOICE_PATTERN.match(pred.strip())):
                fallback_choice = _extract_choice_from_row(row)
                if fallback_choice:
                    pred = fallback_choice

        match = False
        if gold is not None and pred is not None and answers_match(gold, pred):
            match = True
            correct += 1
        else:
            misses.append((idx, gold, pred))

        gold_label = _labelize(gold)
        pred_label = _labelize(pred)
        if match:
            pred_label = gold_label
        labels.append((gold_label, pred_label, match))

    return correct, total, misses, labels


def iter_files(path: Path, recursive: bool) -> Iterable[Path]:
    """Yield JSON/JSONL files from a file or directory."""
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise SystemExit(f"Path not found: {path}")

    pattern = "**/*.json*" if recursive else "*.json*"
    for p in sorted(path.glob(pattern)):
        if p.is_file():
            yield p


def _labels_output_path(src: Path, root: Path, base: Path) -> Path:
    if base.is_dir():
        rel = src.relative_to(base)
        out = root / rel
    else:
        out = root
    return out.with_suffix(".labels.jsonl")


def _write_labels(path: Path, labels: List[Tuple[str, str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for gold_label, pred_label, match in labels:
            handle.write(
                json.dumps(
                    {"gold": gold_label, "pred": pred_label, "correct": match},
                    ensure_ascii=False,
                )
                + "\n"
            )


def report_file(
    path: Path,
    show_misses: bool,
    max_count: int | None,
    dump_labels_root: Path | None,
    base_path: Path,
    include_format_validity: bool,
) -> tuple[int, int, int, int, int]:
    rows = load_rows(path)
    if max_count is not None and max_count > 0:
        rows = rows[:max_count]

    if dump_labels_root is not None:
        correct, total, misses, labels = evaluate_with_labels(rows)
        out_path = _labels_output_path(path, dump_labels_root, base_path)
        _write_labels(out_path, labels)
    else:
        correct, total, misses = evaluate(rows)

    parsed_correct = 0
    parsed_total = 0
    parsed_any = 0
    format_ratio = 0.0
    parse_fail_ratio = 0.0
    if include_format_validity:
        parsed_correct, parsed_total, format_ratio = compute_format_validity(rows)
        parsed_any, total_rows, parsed_ratio = compute_parse_stats(rows)
        parse_fail_ratio = 1.0 - parsed_ratio if total_rows else 0.0

    if total == 0:
        print(f"{path}: no rows to evaluate.")
        return correct, total, parsed_correct, parsed_total, parsed_any

    accuracy = correct / total
    if include_format_validity:
        print(
            f"{path}: {correct}/{total} ({accuracy:.2%}) "
            f"format_validity={format_ratio:.2%} ({parsed_correct}/{parsed_total}) "
            f"parse_fail={parse_fail_ratio:.2%}"
        )
    else:
        print(f"{path}: {correct}/{total} ({accuracy:.2%})")

    if show_misses and misses:
        misses_to_show = misses[: max_count or len(misses)]
        for miss_idx, (idx, gold, pred) in enumerate(misses_to_show, start=1):
            print(f"  Miss {miss_idx} (row {idx}): gold={gold!r}, pred={pred!r}")
    return correct, total, parsed_correct, parsed_total, parsed_any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate answer accuracy for a predictions JSON/JSONL file or directory."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a JSON/JSONL file or a directory containing JSON/JSONL files.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when PATH is a directory.")
    parser.add_argument("--show-misses", action="store_true", help="Print indices of incorrect predictions.")
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Only evaluate the first N rows (per file).",
    )
    parser.add_argument(
        "--dump-labels",
        type=Path,
        default=None,
        help=(
            "Write per-row labels to JSONL. If PATH is a directory (or the input is a "
            "directory), labels are written under PATH preserving relative structure."
        ),
    )
    parser.add_argument(
        "--format-validity",
        action="store_true",
        help="Also report format validity and parse failure rate based on JSON-parsed answers.",
    )
    args = parser.parse_args()

    total_correct = 0
    total_seen = 0
    total_parsed_correct = 0
    total_parsed_total = 0
    total_parsed_any = 0
    files = list(iter_files(args.path, args.recursive))
    if not files:
        hint = " (try --recursive)" if args.path.is_dir() and not args.recursive else ""
        print(f"No JSON/JSONL files found in {args.path}{hint}.")
        return

    dump_root = args.dump_labels
    if dump_root is not None:
        if args.path.is_dir():
            dump_root.mkdir(parents=True, exist_ok=True)
        else:
            if dump_root.is_dir():
                dump_root = dump_root / args.path.name
    for file_path in files:
        c, t, pc, pt, pa = report_file(
            file_path,
            args.show_misses,
            args.max_count,
            dump_root,
            args.path,
            args.format_validity,
        )
        total_correct += c
        total_seen += t
        total_parsed_correct += pc
        total_parsed_total += pt
        total_parsed_any += pa

    if total_seen > 0 and args.path.is_dir():
        overall_acc = total_correct / total_seen
        print(f"Overall: {total_correct}/{total_seen} ({overall_acc:.2%})")
        if args.format_validity:
            overall_format = (
                total_parsed_correct / total_parsed_total if total_parsed_total else 0.0
            )
            overall_parse_fail = (
                1.0 - (total_parsed_any / total_seen) if total_seen else 0.0
            )
            print(
                f"Overall format_validity: {total_parsed_correct}/{total_parsed_total} "
                f"({overall_format:.2%})"
            )
            print(f"Overall parse_fail: {overall_parse_fail:.2%}")
    elif total_seen == 0:
        print("No comparable rows found in the discovered files (missing gold_ans/model_response).")


if __name__ == "__main__":
    main()
