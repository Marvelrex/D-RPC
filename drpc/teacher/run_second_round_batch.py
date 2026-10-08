#!/usr/bin/env python3
"""
Batch process GSM8K questions with second-round prompts, save SFT data, and collect new routes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from drpc.teacher.prompts import (
    build_second_round_system_prompt,
    SECOND_ROUND_MATH_REASONING_PART_THREE_OPTIONS_PROMPT,
    SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_OPTIONS_PROMPT,
    SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_PROMPT,
    SECOND_ROUND_MATH_REASONING_PART_THREE_PROMPT,
)
from drpc.teacher.query_common import extract_response_text, split_answer_and_rationale, _coerce_number
from drpc.bank.router import load_reasoning_bank, route_question
from drpc.bank.update_reasoning_bank import compute_summary, update_reasoning_bank


def _extract_options_text(sample: dict) -> str:
    for key in ("options", "Options", "option", "Option"):
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _append_options(question_text: str, options_text: str) -> str:
    if not options_text:
        return question_text.rstrip()
    base = question_text.rstrip()
    if options_text in base:
        return base
    joiner = "" if base.endswith("\n") else "\n"
    return f"{base}{joiner}{options_text}"


def _aquaize_part_three(text: str, answer_schema: str) -> str:
    """Adjust second-round part_three prompt for AQUA (choices, no missing-path instruction)."""
    if not text:
        return text
    updated = text
    if answer_schema:
        updated = updated.replace("<numeric>", answer_schema)
        updated = updated.replace("<bool>", answer_schema)
        updated = updated.replace('Compute the numeric answer and place it in "ans".', 'Compute the answer and place it in "ans".')
        updated = updated.replace('"ans": <numeric>', f'"ans": {answer_schema}')
        updated = updated.replace('"ans": <bool>', f'"ans": {answer_schema}')
    return updated


def load_questions(path: Path, limit: int, start_index: int = 0) -> List[dict]:
    qs: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            if line_idx < start_index:
                continue
            if len(qs) >= limit:
                break
            obj = json.loads(line)
            qs.append(obj)
    return qs


def load_existing_sft(path: Path) -> List[dict]:
    if not path.exists():
        return []
    raw = None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        try:
            if raw is None:
                raise ValueError("no raw data loaded")
            repaired = raw.rstrip()
            if repaired.endswith(","):
                repaired = repaired[:-1]
            if repaired and repaired[-1] != "]":
                repaired += "\n]"
            data = json.loads(repaired)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except Exception:
            pass
    records = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return records


def persist_sft_records(path: Path, records: List[dict]) -> None:
    """Persist SFT records as a JSON array to keep the file valid (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def base_route_from_bank(route_obj: Dict[str, object]) -> Dict[str, object]:
    cat = route_obj.get("category")
    intent = route_obj.get("intent")
    if isinstance(intent, str):
        intent_list = [intent]
    else:
        intent_list = intent or []
    options = route_obj.get("reasoning_path_options") or []
    reasoning_path = route_obj.get("reasoning_path") or (options[0] if options else [])
    budget = max(len(reasoning_path), 1)
    difficulty = 1 if budget <= 2 else 2
    route = {
        "category": cat,
        "intent": intent_list,
        "difficulty": difficulty,
        "budget": budget,
        "reasoning_path": reasoning_path,
    }
    if options:
        route["reasoning_path_options"] = options
    return route


def _prepare_prompt_route(route: Dict[str, object], top_k_paths: int) -> Dict[str, object]:
    prompt_route = dict(route)
    options = prompt_route.get("reasoning_path_options")
    if isinstance(options, list) and options:
        unique: List[List[str]] = []
        seen = set()
        for option in options:
            if not isinstance(option, list):
                continue
            key = tuple(option)
            if key in seen:
                continue
            seen.add(key)
            unique.append(option)
        trimmed = unique[: max(1, top_k_paths)] if top_k_paths > 0 else unique
        if top_k_paths > 1:
            prompt_route["reasoning_path_options"] = trimmed
            prompt_route.pop("reasoning_path", None)
        else:
            prompt_route.pop("reasoning_path_options", None)
    else:
        prompt_route.pop("reasoning_path_options", None)
    return prompt_route


def call_model(
    client: OpenAI,
    model: str,
    question: str,
    route: dict,
    detailed: bool,
    top_k_paths: int,
    task_type: str,
    is_aqua: bool,
) -> str:
    if task_type == "text":
        system_msg = (
            "You are a structured reasoning tutor.\n"
            "You will be given:\n"
            "- A commonsense reasoning question\n"
            "- One or more routing plans including:\n"
            "    - Category\n"
            "    - Intent\n"
            "    - Budget\n"
            "- ReasoningPathOptions (candidate paths)\n"
            "Pick the best routing plan from options and follow it if adequate; otherwise, refine only the reasoning_path conservatively."
        ).strip()
    else:
        system_msg = build_second_round_system_prompt(top_k_paths).strip()
    use_options = top_k_paths > 1
    if detailed:
        part_three = (
            SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_OPTIONS_PROMPT
            if use_options
            else SECOND_ROUND_MATH_REASONING_PART_THREE_DETAILED_PROMPT
        )
    else:
        part_three = (
            SECOND_ROUND_MATH_REASONING_PART_THREE_OPTIONS_PROMPT
            if use_options
            else SECOND_ROUND_MATH_REASONING_PART_THREE_PROMPT
        )
    if is_aqua:
        part_three = _aquaize_part_three(part_three, "<A|B|C|D|E>")
    if task_type == "text":
        if detailed:
            rationale_block = (
                "2) Generate detailed rationale:\n"
                "   - For each reasoning_path in the chosen reasoning_path, create a detailed rationale string labeled Step1:, Step2:, Step3:, ...\n"
                "   - Use as many substeps as needed; substeps do NOT count toward budget.\n"
                "   - Each substep should include explicit derivation or computation where applicable.\n\n"
            )
            output_rationale = (
                '  "rationale": {\n'
                '    "ReasoningPath1": "Step1: <...> Step2: <...> Step3: <...>",\n'
                '    "ReasoningPath2": "Step1: <...> Step2: <...>"\n'
                "  },\n"
            )
            requirements_tail = "- Do not exceed budget in number of reasoning_path steps.\n- Rationale values must be single strings with Step1:, Step2:, ... labels.\n"
        else:
            rationale_block = (
                "2) Generate rationale:\n"
                "   - For each reasoning_path in the chosen reasoning_path, create a concise reasoning step that include derivation.\n"
                "   - The number of rationale steps must not exceed budget.\n\n"
            )
            output_rationale = (
                '  "rationale": {\n'
                '    "ReasoningPath1": "<...derivation...>",\n'
                '    "ReasoningPath2": "<...derivation...>"\n'
                "  },\n"
            )
            requirements_tail = "- Do not exceed budget in number of steps.\n"
        user_msg = (
            f"Question: {question}\n"
            f"Route: {json.dumps(route, ensure_ascii=False)}\n\n"
            "Task Instructions:\n"
            "1) Assess reasoning_path suitability:\n"
            "   - If the provided reasoning_path can solve the question, keep it.\n"
            "   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.\n"
            "   - Keep category, intent, difficulty, and budget unchanged.\n\n"
            f"{rationale_block}"
            "3) Final answer:\n"
            "   - Compute the boolean answer (true/false) and place it in \"ans\".\n\n"
            "Output (strict JSON, no markdown or extra text):\n"
            "{\n"
            '  "route": { ...same as input, but reasoning_path updated if revised... },\n'
            f"{output_rationale}"
            '  "ans": <bool>\n'
            "}\n\n"
            "Requirements:\n"
            "- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.\n"
            "- Keep reasoning_path names exactly as used in reasoning_path.\n"
            f"{requirements_tail}"
        )
    else:
        user_msg = (
            f"Question: {question}\n"
            f"Route: {json.dumps(route, ensure_ascii=False)}\n\n"
            f"{part_three.strip()}"
        )
    print("---- Prompt Start ----", file=sys.stderr)
    print(system_msg, file=sys.stderr)
    print(user_msg, file=sys.stderr)
    print("---- Prompt End ----", file=sys.stderr)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        seed=42,
        temperature=0.0,
    )
    reply_text = extract_response_text(response)
    print("---- Model Response ----", file=sys.stderr)
    print(reply_text, file=sys.stderr)
    print("-----------------------", file=sys.stderr)
    return reply_text


def _path_in_bank(path: object, base_route: dict) -> bool:
    if not isinstance(path, list) or not path:
        return False
    base_path = base_route.get("reasoning_path")
    if isinstance(base_path, list) and path == base_path:
        return True
    options = base_route.get("reasoning_path_options") or []
    if isinstance(options, list):
        for option in options:
            if isinstance(option, list) and path == option:
                return True
    return False


def _try_load_json(candidate: str) -> dict | None:
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _quote_fractional_ans(payload: str) -> str:
    """Wrap bare fractional ans fields (e.g., 800/9) in quotes to salvage invalid JSON."""
    return re.sub(
        r'"ans"\s*:\s*([0-9]+/[0-9]+)',
        r'"ans": "\1"',
        payload,
        flags=re.MULTILINE,
    )


def parse_model_json(text: str) -> dict | None:
    """Best-effort JSON parsing with fixes for common model formatting issues."""
    attempts = []
    attempts.append(text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        attempts.append(text[start : end + 1].strip())
    if start != -1 and end != -1:
        attempts.append(_quote_fractional_ans(text[start : end + 1]))
    for cand in attempts:
        loaded = _try_load_json(cand)
        if loaded is not None:
            return loaded
    return None


def route_signature(category: str, intent: str, reasoning_path: List[str]) -> tuple:
    return category, intent, tuple(reasoning_path)


def load_existing_new_routes(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [item for item in data if item.get("gold_matches_answer") is True]
    except Exception:
        return []


def append_routes_to_reasoning_bank(reasoning_bank_path: Path, routes: List[dict]) -> int:
    """Append routes into the reasoning bank using update_reasoning_bank, return count added."""
    added = 0
    for item in routes:
        if item.get("gold_matches_answer") is not True:
            continue
        route_obj = item.get("route") or {}
        category = route_obj.get("category")
        intents = route_obj.get("intent") or []
        intent_text = intents[0] if isinstance(intents, list) and intents else intents
        reasoning_path = route_obj.get("reasoning_path") or []
        if category and intent_text and reasoning_path:
            update_reasoning_bank(reasoning_bank_path, category, intent_text, reasoning_path)
            added += 1
    return added


def append_routes_to_record(record_path: Path, routes: List[dict]) -> None:
    """Append all observed routes to a cumulative record (JSON array) without duplicates."""
    if record_path.exists():
        try:
            data = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    signatures = set()
    for item in data:
        r = item.get("route") or {}
        cat = r.get("category")
        intents = r.get("intent") or []
        intent_text = intents[0] if isinstance(intents, list) and intents else intents
        lp = r.get("reasoning_path") or []
        if cat and intent_text and lp:
            signatures.add((cat, intent_text, tuple(lp)))

    for item in routes:
        if item.get("gold_matches_answer") is not True:
            continue
        r = item.get("route") or {}
        cat = r.get("category")
        intents = r.get("intent") or []
        intent_text = intents[0] if isinstance(intents, list) and intents else intents
        lp = r.get("reasoning_path") or []
        sig = (cat, intent_text, tuple(lp))
        if cat and intent_text and lp and sig not in signatures:
            signatures.add(sig)
            data.append(item)

    record_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_reasoning_bank(reasoning_bank_path: Path) -> Dict[str, Dict[str, List[List[str]]]]:
    lb = load_reasoning_bank(reasoning_bank_path)
    return lb["taxonomy"] if "taxonomy" in lb else lb


def write_summary_with_history(summary_path: Path, summary: dict) -> None:
    """Persist summary dict while appending to history with timestamps."""
    base = dict(summary) if summary else {}
    history = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                history = list(existing.get("history", []))
        except Exception:
            history = []
    history.append({"timestamp": datetime.now(timezone.utc).isoformat(), "summary": base})
    out = dict(base)
    out["history"] = history
    summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def save_taxonomy_summary(summary_path: Path, reasoning_bank_path: Path) -> None:
    try:
        rb_obj = load_reasoning_bank(reasoning_bank_path)
        taxonomy = rb_obj.get("taxonomy", rb_obj)
        summary = rb_obj.get("summary")
        if not isinstance(summary, dict) or not summary:
            summary = compute_summary(taxonomy)
        write_summary_with_history(summary_path, summary)
    except Exception as exc:
        print(f"Failed to save taxonomy summary: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run second-round prompts on GSM8K and collect SFT data.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("GSM8K/train.jsonl"),
        help="Path to GSM8K questions (jsonl).",
    )
    parser.add_argument(
        "--reasoning-bank",
        dest="reasoning_bank",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Path to reasoning bank JSON.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=Path("data/reasoning_bank/GSM8K_SFT_Train.json"),
        help="Where to write SFT outputs (JSON).",
    )
    parser.add_argument(
        "--new-routes-output",
        type=Path,
        default=Path("data/reasoning_bank/Unknown_Route.json"),
        help="Where to append newly observed routes (JSON).",
    )
    parser.add_argument(
        "--route-record",
        type=Path,
        default=Path("data/reasoning_bank/Route_Record.json"),
        help="Cumulative log of all observed routes (JSON).",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=500,
        help="Number of questions to process (clamped by dataset length).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Model to query.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Recluster threshold for new routes (informational).",
    )
    parser.add_argument(
        "--top-k-paths",
        type=int,
        default=1,
        help="Number of top reasoning paths to retrieve for the routed intent (default: 1).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "text"],
        default="math",
        help="Prompt flavor (math or text/commonsense).",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use detailed rationale prompt with Step1/Step2 substeps.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based offset into the questions file to start from (default: 0).",
    )
    parser.add_argument(
        "--auto-recluster",
        action="store_true",
        help="Automatically rerun build_taxonomy.py after threshold is reached.",
    )
    parser.add_argument(
        "--filter-similar",
        action="store_true",
        help="After (re)clustering, remove highly similar reasoning paths using word2vec/token-overlap similarity.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold when --filter-similar is set.",
    )
    parser.add_argument(
        "--w2v-model",
        type=Path,
        default=None,
        help="Optional path to a word2vec model (KeyedVectors format). If omitted, falls back to token-overlap similarity.",
    )
    parser.add_argument(
        "--w2v-binary",
        action="store_true",
        help="Set if the provided word2vec model is in binary format.",
    )
    args = parser.parse_args()

    client = OpenAI()
    is_aqua = "aqua" in str(args.questions_path).lower()

    reasoning_bank = refresh_reasoning_bank(args.reasoning_bank)

    existing_routes = set()
    for cat, intents in reasoning_bank.items():
        for intent, paths in intents.items():
            for p in paths:
                existing_routes.add(route_signature(cat, intent, p))

    sft_records: List[dict] = load_existing_sft(args.results_output)
    processed_ids = set()
    for rec in sft_records:
        processed_ids.add(str(rec.get("index")))

    questions = load_questions(args.questions_path, args.num_questions, start_index=args.start_index)

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.new_routes_output.parent.mkdir(parents=True, exist_ok=True)
    args.route_record.parent.mkdir(parents=True, exist_ok=True)

    summary_path = args.reasoning_bank.with_name(args.reasoning_bank.stem + "_summary.json")

    pending_routes: List[dict] = load_existing_new_routes(args.new_routes_output)
    new_routes_seen = len(pending_routes)
    new_routes_records: List[dict] = pending_routes.copy()
    pending_signatures = set()
    for item in new_routes_records:
        route_obj = item.get("route") or {}
        cat = route_obj.get("category")
        intents = route_obj.get("intent") or []
        intent_text = intents[0] if isinstance(intents, list) and intents else intents
        lp = route_obj.get("reasoning_path") or []
        if cat and intent_text and lp:
            pending_signatures.add(route_signature(cat, intent_text, lp))

    processed_count = 0

    def trigger_recluster() -> None:
        nonlocal reasoning_bank, existing_routes, new_routes_records, new_routes_seen, pending_signatures
        added = append_routes_to_reasoning_bank(args.reasoning_bank, new_routes_records)
        append_routes_to_record(args.route_record, new_routes_records)
        print(
            f"Threshold reached (pending routes >= {args.threshold}). "
            f"Appended {added} routes to reasoning bank.",
            file=sys.stderr,
        )
        temp_path = args.new_routes_output.with_suffix(".threshold_snapshot.json")
        temp_path.write_text(json.dumps(new_routes_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Snapshot of pending routes written to {temp_path}", file=sys.stderr)

        if args.auto_recluster:
            tmp_jsonl = args.reasoning_bank.with_name("taxonomy_recluster_input.jsonl")
            lines = []
            lb_obj = load_reasoning_bank(args.reasoning_bank)
            lb_taxonomy = lb_obj.get("taxonomy", lb_obj)
            for cat, intents in lb_taxonomy.items():
                for intent, paths in intents.items():
                    for lp in paths:
                        lines.append({"route": {"category": cat, "intent": [intent], "reasoning_path": lp}, "gold_matches_answer": True})
            for item in new_routes_records:
                r = item.get("route") or {}
                if r:
                    lines.append({"route": r, "gold_matches_answer": True})
            with tmp_jsonl.open("w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")

            from subprocess import run

            helper_build = Path(__file__).resolve().parent.parent / "bank" / "build_taxonomy.py"
            build_cmd = [
                sys.executable,
                str(helper_build),
                "--results-path",
                str(tmp_jsonl),
                "--output-path",
                str(args.reasoning_bank),
            ]
            if args.filter_similar:
                build_cmd += ["--filter-similar", "--sim-threshold", str(args.sim_threshold)]
                if args.w2v_model:
                    build_cmd += ["--w2v-model", str(args.w2v_model)]
                    if args.w2v_binary:
                        build_cmd.append("--w2v-binary")
            print(f"Running recluster: {' '.join(build_cmd)}", file=sys.stderr)
            run(build_cmd, check=False)
            try:
                rb_obj = load_reasoning_bank(args.reasoning_bank)
                summary = rb_obj.get("summary")
                if summary:
                    write_summary_with_history(summary_path, summary)
            except Exception:
                pass

        with args.new_routes_output.open("w", encoding="utf-8") as new_routes_f:
            json.dump([], new_routes_f, ensure_ascii=False, indent=2)
        new_routes_records = []
        pending_signatures = set()
        new_routes_seen = 0
        reasoning_bank = refresh_reasoning_bank(args.reasoning_bank)
        existing_routes = set()
        for cat, intents in reasoning_bank.items():
            for intent, paths in intents.items():
                for p in paths:
                    existing_routes.add(route_signature(cat, intent, p))

    for idx, q in enumerate(questions, start=1):
        question_text = q.get("question", "").strip()
        if not question_text:
            continue
        entry_index = q.get("index")
        if entry_index is None:
            entry_index = idx
        qid = str(q.get("qid") or q.get("id") or q.get("pid") or entry_index)
        if qid in processed_ids or str(entry_index) in processed_ids:
            print(f"Processed question {qid} (already exists)", file=sys.stderr)
            continue

        options_text = _extract_options_text(q) if is_aqua else ""
        question_for_prompt = _append_options(question_text, options_text) if is_aqua else question_text

        routed = route_question(question_for_prompt, reasoning_bank, top_k_paths=max(1, args.top_k_paths))
        base_route = base_route_from_bank(routed)

        prompt_route = _prepare_prompt_route(base_route, max(1, args.top_k_paths))
        raw_reply = call_model(
            client,
            args.model,
            question_for_prompt,
            prompt_route,
            args.detailed,
            max(1, args.top_k_paths),
            args.task_type,
            is_aqua,
        )
        parsed = parse_model_json(raw_reply)
        final_route = base_route
        rationale = None
        ans = None
        reasoning_source = "reasoning_path_bank"

        if isinstance(parsed, dict):
            final_route = parsed.get("route") or base_route
            rationale = parsed.get("rationale")
            ans = parsed.get("ans")
        try:
            final_path = final_route.get("reasoning_path")
            if final_path and not _path_in_bank(final_path, base_route):
                reasoning_source = "llm_generated"
        except Exception:
            pass

        gold_ans_text = q.get("answer")
        if isinstance(gold_ans_text, str):
            _, gold_ans_extracted = split_answer_and_rationale(gold_ans_text)
        else:
            gold_ans_extracted = gold_ans_text
        gold_ans_coerced = _coerce_number(gold_ans_extracted) if gold_ans_extracted is not None else None

        ans_coerced = _coerce_number(ans) if ans is not None else None
        ans_match = False
        try:
            if ans_coerced is not None and gold_ans_coerced is not None:
                ans_match = ans_coerced == gold_ans_coerced
            elif ans is not None and gold_ans_extracted is not None:
                ans_match = str(ans).strip() == str(gold_ans_extracted).strip()
        except Exception:
            ans_match = False

        record: dict = {
            "index": entry_index,
            "qid": qid,
            "question": question_for_prompt,
            "gold_answer": gold_ans_text,
            "gold_answer_extracted": gold_ans_extracted,
            "ans": ans if ans is not None else None,
            "ans_matches_gold": ans_match if ans is not None else False,
            "rationale": rationale,
            "reasoning_path_source": reasoning_source,
            "reasoning_path_bank": base_route,
            "reasoning_path_final": final_route,
            "raw_reply": raw_reply,
            "parse_status": "ok" if rationale is not None and ans is not None else "parse_error",
        }
        sft_records.append(record)
        processed_count += 1

        try:
            cat = final_route.get("category")
            intents = final_route.get("intent") or []
            lp = final_route.get("reasoning_path") or []
            if lp and cat and intents and ans_match:
                intent_text = intents[0] if isinstance(intents, list) else intents
                sig = route_signature(cat, intent_text, lp)
                if sig not in existing_routes and sig not in pending_signatures:
                    existing_routes.add(sig)
                    pending_signatures.add(sig)
                    new_routes_seen += 1
                    new_routes_records.append(
                        {
                            "index": entry_index,
                            "run_index": len(new_routes_records),
                            "route": final_route,
                            "gold_matches_answer": True,
                        }
                    )
        except Exception:
            pass

        print(f"Processed {idx}/{len(questions)} questions", file=sys.stderr)

        if sft_records:
            persist_sft_records(args.results_output, sft_records)
        with args.new_routes_output.open("w", encoding="utf-8") as new_routes_f:
            json.dump(new_routes_records, new_routes_f, ensure_ascii=False, indent=2)

        if processed_count % 500 == 0:
            save_taxonomy_summary(summary_path, args.reasoning_bank)

        if new_routes_seen >= args.threshold:
            trigger_recluster()

    print(f"Completed. Pending new routes: {new_routes_seen}.", file=sys.stderr)

    if new_routes_seen >= args.threshold:
        trigger_recluster()


if __name__ == "__main__":
    main()
