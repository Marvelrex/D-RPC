#!/usr/bin/env python3
"""pass@k evaluation for trained adapters on a JSONL test set.

Reuses drpc/student/distill_rationale.py prompt builders (rpb/cot/freeform/super_correct)
plus a minimal DCoT prompt builder, and drpc/eval/eval_json_accuracy.py for
strategy-agnostic answer parsing. Pass@k is the Chen et al. (2021) unbiased estimator.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


T = TypeVar("T")


def _retry_cuda_load(fn: Callable[[], T], what: str, tries: int = 4, base_sleep: float = 5.0) -> T:
    """Retry a GPU-touching load on transient ``cudaErrorDevicesUnavailable``.

    Compute Canada MIG slices intermittently return device-unavailable when CUDA
    runs early in job startup (NVML enumeration races). The fix used by other
    training jobs is implicit (they touch GPU later via Trainer/accelerate). For
    pass@k we hit the race head-on at adapter load, so wrap explicitly.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except torch.AcceleratorError as e:
            msg = str(e).lower()
            transient = "unavailable" in msg or "busy" in msg or "cuda-capable" in msg
            if not transient or attempt == tries:
                raise
            sleep = base_sleep * attempt
            print(
                f"[RETRY] {what} hit transient CUDA error (attempt {attempt}/{tries}): "
                f"{e!s} — sleeping {sleep}s before retry",
                flush=True,
            )
            last_err = e
            time.sleep(sleep)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    assert last_err is not None
    raise last_err

REPO_ROOT = Path(__file__).resolve().parents[2]

from drpc.student.distill_rationale import (
    _choice_schema_for_path,
    _normalize_task_type,
    build_instructions_with_source,
    build_messages_for_inference,
    format_chat,
    load_jsonl,
)
from drpc.eval.eval_json_accuracy import (
    answers_match,
    extract_gold_ans,
    extract_pred_ans,
    _choice_from_numeric,
    _extract_choice_from_text,
    _map_pred_to_option_choice,
    _extract_choice_from_row,
    CHOICE_PATTERN,
)


def _evaluate_one(row: dict, gold) -> tuple[object, bool]:
    """Mirror of drpc.eval.eval_json_accuracy.evaluate()'s per-row logic.

    The paper's results aggregator (helper/regenerate_new_results_summary.py)
    calls evaluate(), which adds 4 MCQ-specific fallbacks after extract_pred_ans:
      1. numeric → letter via _choice_from_numeric
      2. extract letter from inside free text via _extract_choice_from_text
      3. map pred → option content via _map_pred_to_option_choice
      4. last-ditch _extract_choice_from_row
    Without these, MCQ cells (AI2ARC/AQUA/GPQA) get systematically under-scored
    relative to paper numbers — DCoT/CoT on AQUA loses 10-13 pts.
    """
    pred = extract_pred_ans(row)
    if isinstance(gold, str) and CHOICE_PATTERN.match(gold.strip()):
        numeric_choice = _choice_from_numeric(pred)
        if numeric_choice is not None:
            pred = numeric_choice
        if isinstance(pred, str) and not CHOICE_PATTERN.match(pred.strip()):
            tc = _extract_choice_from_text(pred)
            if tc is not None:
                pred = tc
        if pred is not None and not (isinstance(pred, str) and CHOICE_PATTERN.match(pred.strip())):
            mc = _map_pred_to_option_choice(row, pred)
            if mc:
                pred = mc
        if pred is None or not (isinstance(pred, str) and CHOICE_PATTERN.match(pred.strip())):
            fc = _extract_choice_from_row(row)
            if fc:
                pred = fc
    if gold is None or pred is None:
        return pred, False
    return pred, answers_match(gold, pred)

DCOT_OPTIONS_KEYS = ("options", "choices", "answer_choices")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. (2021) unbiased pass@k estimator."""
    if k > n:
        raise ValueError(f"k ({k}) > n ({n})")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_dcot_prompt(row: dict, num_cots: int) -> str:
    """Replicates acl2025-diverse-cot/scripts/predict_dcot_test.py:build_prompt."""
    parts = []
    q = (row.get("question") or "").strip()
    if q:
        parts.append(f"[Question] {q}")
    ctx = row.get("context")
    if ctx:
        parts.append(f"[Context] {ctx}")
    opts = None
    for key in DCOT_OPTIONS_KEYS:
        if row.get(key):
            opts = row[key]
            break
    if opts:
        if isinstance(opts, list):
            opts = " ".join(str(o).strip() for o in opts)
        parts.append(f"[Options] {opts}")
    parts.append(f"[Number of answers] {max(1, int(num_cots))}")
    parts.append("[Answer 1] ")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter-dir", type=Path, default=None)
    p.add_argument("--no-adapter", action="store_true", help="Skip PEFT adapter; run base model only.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--test-file", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--task-type", default=None)
    p.add_argument("--choice-schema", default=None)
    p.add_argument("--strategy", required=True,
                   choices=["rpb", "cot", "freeform", "super_correct", "dcot", "sgft"])
    p.add_argument("--detailed", action="store_true", default=True,
                   help="Use detailed prompt variant (default: ON; pass --no-detailed to disable).")
    p.add_argument("--no-detailed", dest="detailed", action="store_false")
    p.add_argument("--num-cots", type=int, default=3,
                   help="DCoT only: how many chains the model is asked to produce per completion.")
    p.add_argument("-n", "--num-samples", dest="num_samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-batch-size", type=int, default=8,
                   help="How many of the N samples to generate in parallel per question.")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--output-file", type=Path, required=True,
                   help="JSONL: one row per question with N samples + correctness + (n,c).")
    p.add_argument("--metrics-file", type=Path, default=None,
                   help="JSON summary of pass@k aggregates. Defaults to <output-file>.metrics.json.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume from existing output (default: ON; pass --no-resume to disable).")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--forbid-fresh-when-partial", action="store_true",
                   help="Refuse to overwrite a non-empty output when --no-resume.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--print-prompt", action="store_true", default=True,
                   help="Print first prompt for sanity check (default: ON; pass --no-print-prompt).")
    p.add_argument("--no-print-prompt", dest="print_prompt", action="store_false")
    return p.parse_args()


def build_prompt(row: dict, args: argparse.Namespace, instructions: str, tokenizer) -> str:
    if args.strategy == "dcot":
        return build_dcot_prompt(row, args.num_cots)
    msgs = build_messages_for_inference(row, instructions)
    return format_chat(tokenizer, msgs, add_generation_prompt=True)


def load_completed(output_file: Path) -> dict[str, int]:
    """Map qid -> num samples already on disk."""
    completed: dict[str, int] = {}
    if not output_file.exists():
        return completed
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = str(obj.get("index", obj.get("id", "")))
            if not qid:
                continue
            completed[qid] = max(completed.get(qid, 0), len(obj.get("samples", [])))
    return completed


def main() -> None:
    args = parse_args()
    if args.num_samples < max(args.ks):
        raise SystemExit(f"--num-samples ({args.num_samples}) must be >= max(ks)={max(args.ks)}")

    if not args.resume and args.output_file.exists() and args.output_file.stat().st_size > 0:
        if args.forbid_fresh_when_partial:
            raise SystemExit(
                f"--forbid-fresh-when-partial: {args.output_file} already has data; refusing to clobber."
            )
        args.output_file.unlink()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output_file) if args.resume else {}

    tok_src = args.base_model
    if args.adapter_dir and not args.no_adapter:
        adapter_tok_files = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        if any((Path(args.adapter_dir) / f).is_file() for f in adapter_tok_files):
            tok_src = str(args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    base = _retry_cuda_load(
        lambda: AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=dtype_map[args.dtype], device_map=args.device_map,
        ),
        what=f"base model {args.base_model}",
    )
    if args.adapter_dir and not args.no_adapter:
        model = _retry_cuda_load(
            lambda: PeftModel.from_pretrained(base, args.adapter_dir),
            what=f"adapter {args.adapter_dir}",
        )
        print(f"[INIT] loaded adapter from {args.adapter_dir}", flush=True)
    else:
        model = base
        print("[INIT] running base model with no adapter", flush=True)
    model.eval()

    task_type = args.task_type or _normalize_task_type(None, args.test_file)
    choice_schema = args.choice_schema or _choice_schema_for_path(args.test_file)
    if args.strategy == "dcot":
        instructions, prompt_source = "", "dcot:[Question]/[Options]/[Number of answers]"
    else:
        instructions, prompt_source = build_instructions_with_source(
            args.strategy, detailed=args.detailed, task_type=task_type, choice_schema=choice_schema,
        )
    print(
        f"[INIT] strategy={args.strategy} task_type={task_type} "
        f"choice_schema={choice_schema} prompt_source={prompt_source}",
        flush=True,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_jsonl(args.test_file, limit=args.max_rows)
    _do_sample = args.temperature > 0
    gen_kwargs = dict(
        do_sample=_do_sample,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if _do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    correctness_per_q: list[tuple[int, int]] = []
    written = 0
    printed_prompt = False

    with args.output_file.open("a", encoding="utf-8") as out:
        for q_idx, row in enumerate(rows):
            qid = str(row.get("index", f"test_{q_idx:05d}"))
            need = args.num_samples - completed.get(qid, 0)
            if need <= 0:
                continue

            prompt = build_prompt(row, args, instructions, tokenizer)
            if args.print_prompt and not printed_prompt:
                print(f"\n[PROMPT preview - q{qid}]\n{prompt}\n----\n", flush=True)
                printed_prompt = True

            samples_text: list[str] = []
            B = max(1, args.gen_batch_size)
            for batch_start in range(0, need, B):
                bsz = min(B, need - batch_start)
                inputs = tokenizer(
                    [prompt] * bsz,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_len,
                    padding=True,
                ).to(model.device)
                with torch.no_grad():
                    out_ids = model.generate(**inputs, **gen_kwargs)
                in_len = inputs["input_ids"].shape[-1]
                for j in range(bsz):
                    txt = tokenizer.decode(out_ids[j][in_len:], skip_special_tokens=True).strip()
                    samples_text.append(txt)

            gold = extract_gold_ans(row)
            sample_objs = []
            for txt in samples_text:
                fake_row = dict(row)
                fake_row["model_response"] = txt
                pred, ok = _evaluate_one(fake_row, gold)
                sample_objs.append({"text": txt, "pred": pred, "correct": bool(ok)})

            n = len(sample_objs)
            c = sum(1 for s in sample_objs if s["correct"])
            correctness_per_q.append((n, c))

            out.write(
                json.dumps(
                    {
                        "index": qid,
                        "question": row.get("question"),
                        "gold_ans": gold,
                        "n": n,
                        "c": c,
                        "samples": sample_objs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            written += 1
            if args.log_every and written % args.log_every == 0:
                print(f"[GEN] done={written}/{len(rows)} qid={qid} n={n} c={c}", flush=True)

    aggregated: list[tuple[int, int]] = []
    with args.output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                n = int(obj.get("n", 0))
                c = int(obj.get("c", 0))
            except Exception:
                continue
            if n > 0:
                aggregated.append((n, c))

    metrics: dict[str, object] = {
        "strategy": args.strategy,
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "test_file": str(args.test_file),
        "num_samples_per_question": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "num_questions_evaluated": len(aggregated),
        "num_questions_this_run": len(correctness_per_q),
    }
    for k in args.ks:
        vals = [pass_at_k(n, c, k) for n, c in aggregated if n >= k]
        if vals:
            metrics[f"pass@{k}"] = sum(vals) / len(vals)
            metrics[f"pass@{k}_n_questions"] = len(vals)

    print("\n=== pass@k ===")
    for k_name, v in metrics.items():
        print(f"  {k_name}: {v}")

    metrics_path = args.metrics_file or args.output_file.with_suffix(args.output_file.suffix + ".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\n[METRICS] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
