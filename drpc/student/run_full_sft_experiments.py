#!/usr/bin/env python3
"""
Batch launcher for distill_rationale_full_sft.py with multiple train sizes.

Defaults target RPB rationales for train sizes
[900, 700, 500, 300], writing checkpoints into distilled-<strategy>_ts<train_size>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
SCRIPT = SCRIPTS_DIR / "distill_rationale_full_sft.py"

DEFAULT_TRAIN_SIZES = [900, 700, 500, 300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple full-SFT distillation jobs sequentially.")
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help=f"Train sizes to iterate over (default: {DEFAULT_TRAIN_SIZES}).",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="JSONL dataset to train on (default: based on strategy).",
    )
    parser.add_argument(
        "--strategy",
        type=str.lower,
        default=None,
        help="Strategy to distill (normal, super_correct, RPB, or freeform; defaults to RPB when omitted).",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use detailed rationale prompts when available (structured/freeform strategies).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After each run, generate on the test file (first N rows).",
    )
    parser.add_argument(
        "--max-gen-samples",
        type=int,
        default=300,
        help="Number of test samples to generate when --generate is set (default: 300).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate when --generate is set (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature to pass through to distill_rationale_full_sft.py (default: 0.0).",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling for generation (default: off).",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling for generation (default: off).",
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Override test JSONL path for generation (default: distill_rationale_full_sft.py default).",
    )
    parser.add_argument(
        "--gen-output-file",
        type=Path,
        default=None,
        help="Override output predictions path (default: <output_dir>/predictions.jsonl).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps to pass to distill_rationale_full_sft.py (default: 4).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name/id to pass through (default: meta-llama/Llama-3.2-1B-Instruct).",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer name/id to share across strategies (default: Llama 3.1 8B Instruct tokenizer).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Checkpoint save frequency (steps) passed to distill_rationale_full_sft.py (default: 250).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Logging frequency (steps) passed to distill_rationale_full_sft.py (default: 25).",
    )
    parser.add_argument(
        "--optim",
        type=str,
        choices=["adamw_torch"],
        default="adamw_torch",
        help="Optimizer to pass through to distill_rationale_full_sft.py (restricted to AdamW).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Peak learning rate to pass through to distill_rationale_full_sft.py (default: 1e-4).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio to pass through to distill_rationale_full_sft.py (default: 0.03).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        choices=["linear"],
        default="linear",
        help="LR scheduler used for every run (fixed to linear for uniformity).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm to pass through (shared across runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to reuse for every strategy.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "checkpoints",
        help="Base directory to store model checkpoints.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Base directory to store prediction JSONL files.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=None,
        help="Override model output directory root (defaults to output-root/<model>/<strategy>_ts<size>).",
    )
    parser.add_argument(
        "--prediction-output-dir",
        type=Path,
        default=None,
        help="Override prediction output directory root (defaults to pred-root/<model>/<strategy>_ts<size>).",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=4.0,
        help="Number of epochs to pass through to distill_rationale_full_sft.py (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size to pass through (default: 2).",
    )
    parser.add_argument(
        "--flatten-targets",
        action="store_true",
        help="Flatten rationale/ans fields before training (passed to distill_rationale_full_sft.py).",
    )
    parser.add_argument(
        "--print-chat",
        action="store_true",
        help="Print the formatted chat used for tokenization for every example.",
    )
    parser.add_argument(
        "--cot-shots",
        type=int,
        default=2,
        help="Few-shot count when strategy=cot (default: 2).",
    )
    parser.add_argument(
        "--uniform-signature-file",
        type=Path,
        default=SCRIPTS_DIR / "uniform_training_signature.json",
        help="Shared hyperparameter signature path to enforce parity across runs.",
    )
    parser.add_argument(
        "--reset-uniform-signature",
        action="store_true",
        help="Overwrite the recorded uniform training signature before launching runs.",
    )
    return parser.parse_args()


def count_unique_questions(path: Path) -> int:
    """Count unique question IDs using common id keys; fallback to line index if missing."""
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = [data]
        else:
            iterable = []
    except Exception:
        iterable = None
    if iterable is None:
        iterable = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                iterable.append(obj)
    ids = set()
    for idx, obj in enumerate(iterable):
        if not isinstance(obj, dict):
            continue
        for key in ("index", "id", "question_id", "questionId"):
            if key in obj:
                ids.add(str(obj[key]))
                break
        else:
            ids.add(str(idx))
    return len(ids)


def default_data_file(strategy: str) -> Path:
    raise SystemExit(
        f"No default dataset path for strategy '{strategy}'. Please supply --data-file explicitly."
    )


def build_base_cmd(
    data_file: Path,
    strategy: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--bf16",
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--num-epochs",
        str(args.num_epochs),
        "--data-file",
        str(data_file),
        "--model-name",
        args.model_name,
        "--tokenizer-name",
        args.tokenizer_name,
        "--save-steps",
        str(args.save_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--optim",
        args.optim,
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--lr-scheduler-type",
        args.lr_scheduler_type,
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--seed",
        str(args.seed),
        "--uniform-signature-file",
        str(args.uniform_signature_file.resolve()),
    ]
    if args.detailed:
        cmd.append("--detailed")
    if strategy == "cot":
        cmd += ["--cot-shots", str(args.cot_shots)]
    if strategy:
        cmd += ["--strategy", strategy]
    if args.generate:
        cmd += [
            "--generate",
            "--max-gen-samples",
            str(args.max_gen_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
        ]
        if args.do_sample:
            cmd.append("--do-sample")
        else:
            cmd.append("--no-do-sample")
        if args.test_file:
            cmd += ["--test-file", str(args.test_file)]
        if args.gen_output_file:
            cmd += ["--gen-output-file", str(args.gen_output_file)]
    if args.flatten_targets:
        cmd.append("--flatten-targets")
    if args.print_chat:
        cmd.append("--print-chat")
    if args.reset_uniform_signature:
        cmd.append("--reset-uniform-signature")
    return cmd


def run_experiment(train_size: int, base_cmd: list[str], strategy: str, args: argparse.Namespace, single_run: bool, dataset_size: int) -> None:
    effective_size = min(train_size, dataset_size)
    if effective_size < train_size:
        print(f"[WARN] Requested train_size={train_size} exceeds dataset_size={dataset_size}; capping to {effective_size}", flush=True)
    suffix = strategy
    strat_suffix = "_detailed" if args.detailed else ""
    model_name_arg = Path(base_cmd[base_cmd.index("--model-name") + 1]).name

    if args.model_output_dir:
        if single_run:
            output_dir = args.model_output_dir
        else:
            output_dir = args.model_output_dir / f"{suffix}{strat_suffix}_ts{train_size}"
    else:
        output_dir = args.output_root / f"{model_name_arg}" / f"{suffix}{strat_suffix}_ts{train_size}"
    if effective_size != train_size:
        output_dir = args.output_root / f"{model_name_arg}" / f"{suffix}{strat_suffix}_ts{effective_size}"

    if args.prediction_output_dir:
        run_pred_dir = args.prediction_output_dir if single_run else args.prediction_output_dir / f"{suffix}{strat_suffix}_ts{effective_size}"
    else:
        run_pred_dir = args.pred_root / f"{model_name_arg}" / f"{suffix}{strat_suffix}_ts{effective_size}"
    run_pred_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = (args.test_file or Path("test")).stem
    default_pred = run_pred_dir / f"{suffix}{strat_suffix}_{model_name_arg}_{dataset_label}_{effective_size}.jsonl"

    cmd = list(base_cmd)
    if args.generate and not args.gen_output_file:
        cmd += ["--gen-output-file", str(default_pred)]
    cmd += [
        "--train-size",
        str(effective_size),
        "--output-dir",
        str(output_dir),
    ]
    detail_note = " (detailed)" if args.detailed else ""
    print(f"\n=== Running strategy={strategy}{detail_note} train_size={effective_size} (requested {train_size}) -> {output_dir} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    strategy = (args.strategy or "rpb").lower()
    valid = {"normal", "super_correct", "rpb", "freeform", "cot"}
    if strategy not in valid:
        raise SystemExit(f"--strategy must be one of {sorted(valid)}")

    data_path = args.data_file or default_data_file(strategy)
    dataset_size = count_unique_questions(data_path)
    if dataset_size <= 0:
        raise SystemExit(f"No data found in {data_path} to build training set.")
    base_cmd = build_base_cmd(data_path, strategy, args)
    single_run = len(args.train_sizes) == 1
    for size in args.train_sizes:
        run_experiment(size, base_cmd, strategy, args, single_run, dataset_size)


if __name__ == "__main__":
    main()
