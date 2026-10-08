#!/usr/bin/env python3
"""
Train a word2vec model on taxonomy reasoning paths and save it in KeyedVectors format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from drpc.bank.taxonomy_tokens import split_tokens
from gensim.models import Word2Vec


def build_sentences(taxonomy_path: Path) -> List[List[str]]:
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy = data.get("taxonomy", data)
    sentences: List[List[str]] = []
    for intents in taxonomy.values():
        for paths in intents.values():
            for path in paths:
                toks: List[str] = []
                for step in path:
                    toks.extend(split_tokens(step))
                if toks:
                    sentences.append(toks)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Train word2vec on taxonomy reasoning paths.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Path to taxonomy JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy_word2vec.bin"),
        help="Where to write the trained KeyedVectors model (binary format).",
    )
    parser.add_argument("--vector-size", type=int, default=100, help="Embedding dimension (default: 100).")
    parser.add_argument("--window", type=int, default=5, help="Context window size (default: 5).")
    parser.add_argument("--min-count", type=int, default=1, help="Min token frequency (default: 1).")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    if not args.taxonomy.exists():
        raise SystemExit(f"Taxonomy file not found: {args.taxonomy}")

    sentences = build_sentences(args.taxonomy)
    if not sentences:
        raise SystemExit("No tokens extracted from taxonomy; aborting.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save_word2vec_format(str(args.output), binary=True)
    print(f"Saved word2vec model to {args.output}")


if __name__ == "__main__":
    main()
