#!/usr/bin/env python3
"""Regenerate the bank-ablation teacher data (empty / random / randglobal).

RECONSTRUCTION: the original generator was not preserved. This script re-implements it from the
released data files (the exact system/user prompt stored in every record is reproduced verbatim)
and the recipes used when the data were created:
  empty       the route carries no candidate path (reasoning_path_options = [])
  random      the retrieved path is replaced by a path drawn from an item with a different
              (category, intent), truncated to the item's budget
  randglobal  the retrieved path is replaced by a uniform draw from the whole bank, excluding
              the item's own retrieved paths
The teacher (default gpt-5.1) then writes the rationale exactly as in the full D-RPC pipeline.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drpc.eval.eval_json_accuracy import answers_match  # noqa: E402

SYSTEM_PROMPT = (
    "You are a structured reasoning tutor.\n"
    "You will be given:\n"
    "- A math question\n"
    "- One or more routing plans including:\n"
    "    - Category\n"
    "    - Intent\n"
    "    - Budget\n"
    "- ReasoningPathOptions (candidate paths)\n"
    "Pick the best routing plan from options and follow it if it is adequate; otherwise, refine only the reasoning_path conservatively."
)

USER_SUFFIX = {
    "numeric": (
        "\n\nTask Instructions:\n\n"
        "1) Assess reasoning_path suitability:\n"
        "   - If the provided reasoning_path can solve the question, keep it.\n"
        "   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.\n"
        "   - Keep category, intent, difficulty, and budget unchanged.\n\n"
        "2) Generate detailed rationale:\n"
        "   - For each reasoning_path in the chosen reasoning_path, create a detailed rationale string labeled Step1:, Step2:, Step3:, ...\n"
        "   - Use as many substeps as needed; substeps do NOT count toward budget.\n"
        "   - Each substep should include explicit derivation or computation where applicable.\n\n"
        "3) Final answer:\n"
        "   - Compute the numeric answer and place it in \"ans\".\n\n"
        "Output (strict JSON, no markdown or extra text):\n"
        "{\n"
        "  \"route\": { ...same as input, but reasoning_path updated if revised... },\n"
        "  \"rationale\": {\n"
        "    \"ReasoningPath1\": \"Step1: <...> Step2: <...>\",\n"
        "    \"ReasoningPath2\": \"Step1: <...> Step2: <...>\"\n"
        "  },\n"
        "  \"ans\": <numeric>\n"
        "}\n\n"
        "Requirements:\n"
        "- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.\n"
        "- Keep reasoning_path names exactly as used in reasoning_path.\n"
        "- Do not exceed budget in number of reasoning_path steps.\n"
        "- Rationale values must be single strings with Step1:, Step2:, ... labels."
    ),
    "choice": (
        "\n\nTask Instructions:\n\n"
        "1) Assess reasoning_path suitability:\n"
        "   - If the provided reasoning_path can solve the question, keep it.\n"
        "   - If not, create a revised reasoning_path (TitleCase, no question-specific wording, no digits/underscores) with length <= budget.\n"
        "   - Keep category, intent, difficulty, and budget unchanged.\n\n"
        "2) Generate detailed rationale:\n"
        "   - For each reasoning_path in the chosen reasoning_path, create a detailed rationale string labeled Step1:, Step2:, Step3:, ...\n"
        "   - Use as many substeps as needed; substeps do NOT count toward budget.\n"
        "   - Each substep should include explicit derivation or computation where applicable.\n\n"
        "3) Final answer:\n"
        "   - Compute the answer and place it in \"ans\".\n\n"
        "Output (strict JSON, no markdown or extra text):\n"
        "{\n"
        "  \"route\": { ...same as input, but reasoning_path updated if revised... },\n"
        "  \"rationale\": {\n"
        "    \"ReasoningPath1\": \"Step1: <...> Step2: <...>\",\n"
        "    \"ReasoningPath2\": \"Step1: <...> Step2: <...>\"\n"
        "  },\n"
        "  \"ans\": <A|B|C|D|E>\n"
        "}\n\n"
        "Requirements:\n"
        "- Each reasoning_path in route.reasoning_path must appear exactly once as a key in rationale, in order.\n"
        "- Keep reasoning_path names exactly as used in reasoning_path.\n"
        "- Do not exceed budget in number of reasoning_path steps.\n"
        "- Rationale values must be single strings with Step1:, Step2:, ... labels."
    ),
}
ROUTE_KEYS = ["category", "intent", "difficulty", "budget", "reasoning_path_options"]


def build_user_prompt(question: str, route: dict, answer_type: str) -> str:
    return f"Question: {question}\nRoute: {json.dumps(route)}" + USER_SUFFIX[answer_type]


def perturb(item: dict, mode: str, bank: list, rng: random.Random) -> dict:
    src = item["reasoning_path_bank"]
    own = tuple(src.get("reasoning_path") or [])
    bank_entry = {k: src.get(k) for k in ["category", "intent", "difficulty", "budget"]}
    if mode == "empty":
        path = []
    elif mode == "random":
        key = (src.get("category"), tuple(src.get("intent") or []))
        pool = [p for p, meta in bank if meta != key] or [p for p, _ in bank]
        path = list(rng.choice(pool))[: int(src.get("budget") or len(own) or 1)]
    elif mode == "randglobal":
        pool = [p for p, _ in bank if p != own] or [p for p, _ in bank]
        path = list(rng.choice(pool))
    else:
        raise ValueError(mode)
    bank_entry["reasoning_path"] = path
    bank_entry["reasoning_path_options"] = [path] if path else []
    return bank_entry


def parse_reply(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1]), "ok"
            except json.JSONDecodeError:
                pass
    return None, "parse_error"


def call_teacher(client, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="full D-RPC second-round file (routes + bank paths)")
    parser.add_argument("--mode", choices=["empty", "random", "randglobal"], required=True)
    parser.add_argument("--answer-type", choices=["numeric", "choice"], required=True, help="numeric for MATH, choice for AQUA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="write prompts only, do not call the teacher")
    args = parser.parse_args()

    items = json.load(open(args.source))
    if args.limit:
        items = items[: args.limit]
    rng = random.Random(args.seed)
    bank = []
    seen = set()
    for it in items:
        b = it["reasoning_path_bank"]
        p = tuple(b.get("reasoning_path") or [])
        if p and p not in seen:
            seen.add(p)
            bank.append((p, (b.get("category"), tuple(b.get("intent") or []))))

    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI()

    done = {}
    if args.output.exists():
        for rec in json.load(open(args.output)):
            done[rec["index"]] = rec
    out = []
    for it in items:
        if it["index"] in done:
            out.append(done[it["index"]])
            continue
        bank_entry = perturb(it, args.mode, bank, rng)
        route = {k: bank_entry[k] for k in ROUTE_KEYS}
        user_prompt = build_user_prompt(it["question"], route, args.answer_type)
        rec = {k: it[k] for k in ["index", "qid", "question", "gold_answer", "gold_answer_extracted"] if k in it}
        rec.update(reasoning_path_bank=bank_entry, model=args.model,
                   prompt=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])
        if args.dry_run:
            out.append(rec)
            continue
        raw = call_teacher(client, args.model, SYSTEM_PROMPT, user_prompt)
        parsed, status = parse_reply(raw)
        reply_route = (parsed or {}).get("route") if isinstance(parsed, dict) else None
        final = dict(reply_route) if isinstance(reply_route, dict) else {k: bank_entry[k] for k in ROUTE_KEYS}
        revised = isinstance(reply_route, dict) and "reasoning_path" in reply_route and list(reply_route["reasoning_path"]) != bank_entry["reasoning_path"]
        ans = (parsed or {}).get("ans") if isinstance(parsed, dict) else None
        gold = rec.get("gold_answer_extracted") or rec.get("gold_answer")
        rec.update(
            ans=ans,
            ans_matches_gold=bool(gold is not None and ans is not None and answers_match(str(gold), str(ans))),
            rationale=(parsed or {}).get("rationale") if isinstance(parsed, dict) else None,
            reasoning_path_source="llm_generated" if revised else "reasoning_path_bank",
            reasoning_path_final=final,
            raw_reply=raw,
            parse_status=status,
        )
        out.append(rec)
        if len(out) % 50 == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(args.output, "w"), ensure_ascii=False, indent=1)
            print(f"{len(out)}/{len(items)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} records to {args.output}")


if __name__ == "__main__":
    main()
