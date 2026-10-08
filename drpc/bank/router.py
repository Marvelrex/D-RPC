#!/usr/bin/env python3
"""
Semantic router: map a math question to (Category, Intent, ReasoningPath) from a reasoning bank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit("Install requirements: pip install sentence-transformers") from exc


def load_reasoning_bank(path: Path) -> Dict[str, Dict[str, List[List[str]]]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pick_topk_reasoning_paths(
    question_emb: np.ndarray,
    reasoning_paths: List[List[str]],
    embed: callable,
    top_k: int,
) -> List[List[str]]:
    """Return the top_k reasoning paths ranked by cosine similarity to the question."""
    scored = []
    for path in reasoning_paths:
        text = " ".join(path)
        path_emb = embed([text])[0]
        score = cosine_similarity(question_emb, path_emb)
        scored.append((score, path))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[: max(1, top_k)]]


def route_question(
    question: str,
    reasoning_bank: Dict[str, Dict[str, List[List[str]]]],
    model: SentenceTransformer | None = None,
    top_k_paths: int = 1,
) -> Dict[str, object]:
    model = model or SentenceTransformer("all-MiniLM-L6-v2")
    embed = lambda texts: model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    q_emb = embed([question])[0]

    best_pair: Tuple[str, str] | None = None
    best_score = -1.0

    for category, intents in reasoning_bank.items():
        for intent_text in intents.keys():
            intent_emb = embed([intent_text])[0]
            score = cosine_similarity(q_emb, intent_emb)
            if score > best_score:
                best_score = score
                best_pair = (category, intent_text)

    if best_pair is None:
        return {"category": None, "intent": None, "reasoning_path": []}

    category, intent_text = best_pair
    reasoning_paths = reasoning_bank[category][intent_text]
    top_paths = pick_topk_reasoning_paths(q_emb, reasoning_paths, embed, top_k=top_k_paths)
    best_reasoning_path = top_paths[0] if top_paths else []

    return {
        "category": category,
        "intent": intent_text,
        "reasoning_path": best_reasoning_path,
        "reasoning_path_options": top_paths,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Route a question to (Category, Intent, ReasoningPath).")
    parser.add_argument("question", type=str, help="Natural language math question.")
    parser.add_argument(
        "--reasoning-bank",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Path to reasoning bank JSON (expects top-level keys categories, intents, and paths).",
    )
    args = parser.parse_args()

    data = load_reasoning_bank(args.reasoning_bank)
    if "taxonomy" in data:
        data = data["taxonomy"]

    result = route_question(args.question, data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
