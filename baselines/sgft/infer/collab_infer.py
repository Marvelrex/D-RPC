#!/usr/bin/env python3
"""
Collaborative Inference for SGFT (arXiv:2412.09906v1, §3.2).

Pipeline per question:
  1. Guide model  → generate Solution Guidance (SG) using prompt_iv (temp=0)
  2. Validate SG (2–6 steps, no calculations); retry once if invalid
  3. Response model → generate final answer using Question + SG (temp=0)
  4. Extract answer per dataset (GSM8K=numeric, AI2ARC/AQUA/GPQA=A-E, StrategyQA=yes/no)

Supported combos (--guide-model / --response-model can differ or be the same):
  1. guide=Qwen-3-1.7B,   response=Qwen-3-1.7B
  2. guide=Qwen-3-1.7B,   response=Llama-3.1-8B
  3. guide=Llama-3.1-8B,  response=Llama-3.1-8B
  4. guide=Llama-3.1-8B,  response=Qwen-3-1.7B

Temperature is fixed at 0 throughout (paper §3.2 deterministic setting).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA_PREP = _HERE.parent / "data_prep"
if str(_DATA_PREP) not in sys.path:
    sys.path.insert(0, str(_DATA_PREP))

from clean_sg import validate_sg


PROMPT_IV = (
    "Please generate a step-by-step solution for the following problem with no calculations.\n"
    " You don't need to solve it, just output the steps in 2 to 6 steps."
)

RETRY_CONSTRAINT = (
    "Do not include any calculations or numeric operations; only describe high-level steps."
)

RESPONSE_PROMPT_TEMPLATE = (
    "Question: {question}\nSolution Guidance:\n{sg}\nAnswer:"
)


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the last standalone number from model output."""
    text = text.strip()
    if "####" in text:
        parts = text.split("####")
        candidate = parts[-1].strip()
        nums = re.findall(r"-?\d+(?:[,\d]*)?(?:\.\d+)?", candidate)
        if nums:
            return nums[-1].replace(",", "")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "ans" in payload:
            v = str(payload["ans"]).strip()
            if re.fullmatch(r"-?\d+(?:\.\d+)?", v):
                return v
    except Exception:
        pass
    nums = re.findall(r"-?\d+(?:[,\d]*)?(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _extract_aqua_answer(text: str) -> Optional[str]:
    """Extract the last A-E letter from model output."""
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "ans" in payload:
            v = str(payload["ans"]).strip().upper()
            if v in "ABCDE" and len(v) == 1:
                return v
    except Exception:
        pass
    for pattern in [
        r"answer\s+is\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        r"(?:^|[\s\(\[])\b([A-E])\b(?:\)|]|\s|$)",
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def _extract_strategyqa_answer(text: str) -> Optional[str]:
    """Extract yes/no from model output."""
    text_lower = text.strip().lower()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "ans" in payload:
            v = str(payload["ans"]).strip().lower()
            if v in ("yes", "true", "1"):
                return "yes"
            if v in ("no", "false", "0"):
                return "no"
    except Exception:
        pass
    tokens = re.findall(r"\b(yes|no|true|false)\b", text_lower)
    if tokens:
        t = tokens[-1]
        return "yes" if t in ("yes", "true") else "no"
    return None


EXTRACTORS = {
    "gsm8k": _extract_gsm8k_answer,
    "strategyqa": _extract_strategyqa_answer,
    "aqua": _extract_aqua_answer,
    "ai2arc": _extract_aqua_answer,
    "gpqa": _extract_aqua_answer,
}


def _load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
) -> tuple:
    """Load a HuggingFace model + tokenizer; handles PEFT/LoRA checkpoints."""
    ckpt = Path(model_path)
    adapter_cfg = ckpt / "adapter_config.json"

    if adapter_cfg.exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError:
            raise ImportError("Install peft: pip install peft")
        cfg = PeftConfig.from_pretrained(str(ckpt))
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path,
            device_map="auto",
            dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt))
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_name_or_path, use_fast=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_gen_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen_prompt
        )
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user = messages[1]["content"] if len(messages) > 1 and messages[1]["role"] == "user" else ""
    if add_gen_prompt:
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    asst = messages[2]["content"] if len(messages) > 2 and messages[2]["role"] == "assistant" else ""
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {asst}</s>"


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Run greedy decode (temperature=0) on a single prompt."""
    eos_id = tokenizer.eos_token_id
    eot_id: Optional[int] = None
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        pass
    eos_list: List[int] = []
    if isinstance(eos_id, list):
        eos_list.extend(eos_id)
    elif eos_id is not None:
        eos_list.append(eos_id)
    if eot_id is not None and eot_id not in eos_list and eot_id != -1:
        eos_list.append(eot_id)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_list or eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def run_collab_infer(
    guide_model,
    guide_tok,
    response_model,
    response_tok,
    test_rows: List[dict],
    dataset: str,
    output_path: Path,
    max_new_tokens_sg: int = 256,
    max_new_tokens_ans: int = 512,
    combo_id: str = "unknown",
    guide_ckpt: str = "",
) -> None:
    """Run the guide→SG→response pipeline and stream results to output_path."""
    extractor = EXTRACTORS.get(dataset)
    if extractor is None:
        raise ValueError(f"No extractor for dataset '{dataset}'. Options: {list(EXTRACTORS)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if "index" in obj:
                        done_ids.add(str(obj["index"]))
                except Exception:
                    continue
        if done_ids:
            print(f"[INFER] Resuming — {len(done_ids)} examples already done.", flush=True)

    with output_path.open("a", encoding="utf-8") as fh:
        for row_idx, row in enumerate(test_rows):
            idx = str(row.get("index", f"test_{row_idx:05d}"))
            if idx in done_ids:
                continue

            question = row.get("question", "").strip()
            options_text = row.get("options", "")
            if options_text and options_text not in question:
                question = f"{question}\n{options_text}"

            guide_user = f"{PROMPT_IV}\n\n{question}"
            guide_messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": guide_user},
            ]
            guide_prompt = _apply_chat_template(guide_tok, guide_messages, add_gen_prompt=True)
            sg = _generate(guide_model, guide_tok, guide_prompt, max_new_tokens=max_new_tokens_sg)

            sg_valid, sg_reasons = validate_sg(sg)
            sg_retry_used = False
            if not sg_valid:
                retry_user = guide_user + "\n" + RETRY_CONSTRAINT
                retry_messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": retry_user},
                ]
                retry_prompt = _apply_chat_template(
                    guide_tok, retry_messages, add_gen_prompt=True
                )
                sg = _generate(
                    guide_model, guide_tok, retry_prompt, max_new_tokens=max_new_tokens_sg
                )
                sg_valid, sg_reasons = validate_sg(sg)
                sg_retry_used = True

            response_user = RESPONSE_PROMPT_TEMPLATE.format(
                question=question, sg=sg
            )
            response_messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": response_user},
            ]
            response_prompt = _apply_chat_template(
                response_tok, response_messages, add_gen_prompt=True
            )
            answer_text = _generate(
                response_model, response_tok, response_prompt, max_new_tokens=max_new_tokens_ans
            )

            model_ans = extractor(answer_text)
            gold_ans = row.get("answer", row.get("gold_ans", row.get("ans", "")))
            if isinstance(gold_ans, bool):
                gold_ans = "yes" if gold_ans else "no"
            gold_ans = str(gold_ans).strip()

            out_row = {
                "dataset": dataset,
                "index": idx,
                "question": row.get("question", "").strip(),
                "gold_ans": gold_ans,
                "sg": sg,
                "sg_valid": sg_valid,
                "sg_reasons": [r.value for r in sg_reasons],
                "sg_retry_used": sg_retry_used,
                "answer_raw": answer_text,
                "model_ans": model_ans,
                "combo_id": combo_id,
                "guide_ckpt": guide_ckpt,
            }
            fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fh.flush()

            status = "✓" if model_ans and model_ans.lower() == gold_ans.lower() else "✗"
            print(
                f"  [{row_idx+1}] {idx} | SG_ok={sg_valid} | ans={model_ans!r} gold={gold_ans!r} {status}",
                flush=True,
            )


def load_eval_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collaborative inference: guide model generates SG, response model answers."
    )
    p.add_argument("--guide-ckpt", required=True, help="Path to SGFT guide model checkpoint.")
    p.add_argument(
        "--response-model",
        required=True,
        help="HF model id or path for the response model (can equal --guide-ckpt).",
    )
    p.add_argument(
        "--dataset",
        choices=["gsm8k", "strategyqa", "aqua", "ai2arc", "gpqa"],
        required=True,
    )
    p.add_argument("--test-file", type=Path, required=True, help="Eval split JSONL.")
    p.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSONL path (default: auto-named in Baselines/SGFT/outputs/).",
    )
    p.add_argument(
        "--max-gen-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N rows (default: all).",
    )
    p.add_argument("--max-new-tokens-sg", type=int, default=256)
    p.add_argument("--max-new-tokens-ans", type=int, default=512)
    p.add_argument(
        "--combo-id",
        default=None,
        help="Human-readable label for this guide/response combo (e.g. qwen-qwen).",
    )
    p.add_argument("--bf16", action="store_true", help="Load models in bfloat16.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if args.bf16 and has_cuda else (
        torch.float16 if has_cuda else torch.float32
    )

    guide_ckpt = args.guide_ckpt
    response_model_id = args.response_model
    same_model = (guide_ckpt == response_model_id)

    combo_id = args.combo_id or f"guide={Path(guide_ckpt).name}_resp={Path(response_model_id).name}"

    if args.output_file:
        output_path = args.output_file
    else:
        out_dir = Path("Baselines/SGFT/outputs") / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", combo_id)
        output_path = out_dir / f"predictions_{safe}.jsonl"

    print(f"[COLLAB] dataset={args.dataset}  combo={combo_id}", flush=True)
    print(f"[COLLAB] guide   = {guide_ckpt}", flush=True)
    print(f"[COLLAB] response= {response_model_id}", flush=True)
    print(f"[COLLAB] output  = {output_path}", flush=True)

    print("\n[1/3] Loading guide model …", flush=True)
    guide_model, guide_tok = _load_model_and_tokenizer(guide_ckpt, dtype)

    if same_model:
        print("[2/3] Response model = guide model (shared).", flush=True)
        response_model, response_tok = guide_model, guide_tok
    else:
        print("\n[2/3] Loading response model …", flush=True)
        response_model, response_tok = _load_model_and_tokenizer(response_model_id, dtype)

    print("\n[3/3] Running inference …", flush=True)
    test_rows = load_eval_jsonl(args.test_file, limit=args.max_gen_samples)
    print(f"  Test rows: {len(test_rows)}", flush=True)

    run_collab_infer(
        guide_model=guide_model,
        guide_tok=guide_tok,
        response_model=response_model,
        response_tok=response_tok,
        test_rows=test_rows,
        dataset=args.dataset,
        output_path=output_path,
        max_new_tokens_sg=args.max_new_tokens_sg,
        max_new_tokens_ans=args.max_new_tokens_ans,
        combo_id=combo_id,
        guide_ckpt=guide_ckpt,
    )
    print(f"\n[COLLAB] Done. Predictions saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
