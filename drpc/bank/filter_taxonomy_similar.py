#!/usr/bin/env python3
"""
Filter taxonomy reasoning paths by collapsing highly similar paths.

Similarity uses word2vec embeddings when a model is provided; otherwise it
falls back to simple token-overlap Jaccard similarity. The filtered taxonomy
is written to a new JSON file so the original stays untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from gensim.models import KeyedVectors
except ImportError:
    KeyedVectors = None

from drpc.bank.taxonomy_tokens import split_tokens


def tokenize_path(path_steps: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for step in path_steps:
        tokens.extend(split_tokens(step))
    return tokens


class SimilarityComputer:
    """Compute similarity between reasoning paths with optional word2vec support."""

    def __init__(self, kv: Optional["KeyedVectors"]) -> None:
        self.kv = kv
        self.cache: Dict[Tuple[str, ...], Tuple[Optional[np.ndarray], Set[str]]] = {}

    def _vectorize(self, tokens: List[str]) -> Optional[np.ndarray]:
        if self.kv is None:
            return None
        vecs = []
        for tok in tokens:
            if tok in self.kv:
                vecs.append(self.kv[tok])
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    def get_repr(self, path: Sequence[str]) -> Tuple[Optional[np.ndarray], Set[str]]:
        key = tuple(path)
        if key not in self.cache:
            tokens = tokenize_path(path)
            self.cache[key] = (self._vectorize(tokens), set(tokens))
        return self.cache[key]

    def similarity(self, a: Sequence[str], b: Sequence[str]) -> float:
        vec_a, tokens_a = self.get_repr(a)
        vec_b, tokens_b = self.get_repr(b)

        if vec_a is not None and vec_b is not None:
            denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            if denom > 0.0:
                return float(np.dot(vec_a, vec_b) / denom)
            return 0.0

        if not tokens_a and not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def filter_paths(
    paths: List[List[str]],
    sim: SimilarityComputer,
    threshold: float,
) -> Tuple[List[List[str]], List[Tuple[List[str], float, List[str]]]]:
    kept: List[List[str]] = []
    removed: List[Tuple[List[str], float, List[str]]] = []
    for path in paths:
        best = 0.0
        best_path: List[str] | None = None
        for kept_path in kept:
            val = sim.similarity(path, kept_path)
            if val > best:
                best = val
                best_path = kept_path
            if best >= threshold:
                break
        if best >= threshold:
            removed.append((path, best, best_path or []))
        else:
            kept.append(path)
    return kept, removed


def filter_taxonomy(
    taxonomy: Dict[str, Dict[str, List[List[str]]]],
    sim: SimilarityComputer,
    threshold: float,
) -> Tuple[Dict[str, Dict[str, List[List[str]]]], Dict[str, int], List[Tuple[str, str, List[str], List[str], float]]]:
    filtered: Dict[str, Dict[str, List[List[str]]]] = {}
    removed_count = 0
    unchanged = 0
    removed_pairs: List[Tuple[str, str, List[str], List[str], float]] = []

    for category, intents in taxonomy.items():
        filtered[category] = {}
        for intent, paths in intents.items():
            kept, removed = filter_paths(paths, sim, threshold)
            filtered[category][intent] = kept
            for path, score, kept_path in removed:
                removed_pairs.append((category, intent, path, kept_path, score))
            removed_count += len(removed)
            if len(removed) == 0:
                unchanged += 1

    stats = {
        "removed_paths": removed_count,
        "unchanged_buckets": unchanged,
    }
    return filtered, stats, removed_pairs


def recompute_summary(
    taxonomy: Dict[str, Dict[str, List[List[str]]]],
    raw_intents_before_clustering: Optional[int] = None,
) -> Dict[str, int]:
    categories = len(taxonomy)
    intents: Set[str] = set()
    reasoning_paths: Set[Tuple[str, ...]] = set()
    for intent_map in taxonomy.values():
        intents.update(intent_map.keys())
        for paths in intent_map.values():
            for path in paths:
                reasoning_paths.add(tuple(path))
    summary = {
        "unique_categories": categories,
        "unique_intents": len(intents),
        "unique_reasoning_paths": len(reasoning_paths),
        "unique_canonical_intents": len(intents),
        "raw_intents_before_clustering": raw_intents_before_clustering
        if raw_intents_before_clustering is not None
        else len(intents),
    }
    return summary


def load_word2vec(model_path: Optional[Path], binary: bool) -> Optional["KeyedVectors"]:
    if model_path is None:
        print("No word2vec model supplied; falling back to token overlap similarity.", file=sys.stderr)
        return None
    if KeyedVectors is None:
        raise SystemExit("gensim is required to load word2vec models. Install with `pip install gensim`.")
    print(f"Loading word2vec model from {model_path} (binary={binary})...", file=sys.stderr)
    return KeyedVectors.load_word2vec_format(str(model_path), binary=binary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapse highly similar taxonomy reasoning paths using word2vec similarity.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Path to the taxonomy JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a word2vec model (KeyedVectors format). When omitted, falls back to token-overlap similarity.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Set if the provided word2vec model is in binary format.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold above which a path is considered redundant and removed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the filtered taxonomy (default: <taxonomy>-filtered.json alongside the input).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts; do not write an output file.",
    )
    parser.add_argument(
        "--show-removed",
        action="store_true",
        help="Print removed paths with the kept path and similarity score.",
    )
    args = parser.parse_args()

    if not args.taxonomy.exists():
        raise SystemExit(f"Taxonomy file not found: {args.taxonomy}")

    model = load_word2vec(args.model_path, args.binary)
    sim = SimilarityComputer(model)

    data = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    taxonomy = data.get("taxonomy")
    if not isinstance(taxonomy, dict):
        raise SystemExit("Unexpected taxonomy format; expected a top-level 'taxonomy' dict.")

    filtered_taxonomy, stats, removed_pairs = filter_taxonomy(taxonomy, sim, args.threshold)
    summary = recompute_summary(filtered_taxonomy)

    removed = stats["removed_paths"]
    if removed == 0:
        print("No paths were removed at the chosen threshold.")
    else:
        print(f"Removed {removed} paths at similarity >= {args.threshold}.")
    print(f"Resulting summary: {summary}")

    if args.show_removed:
        if not removed_pairs:
            print("No removed paths to display.")
        else:
            print("Removed paths (path -> kept path @ similarity):")
            for category, intent, path, kept_path, score in removed_pairs:
                print(f"  {category}/{intent}: {path}  ~~({score:.2f})~~>  {kept_path}")

    if args.dry_run:
        return

    output = args.output
    if output is None:
        output = args.taxonomy.with_name(args.taxonomy.stem + "_filtered.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"taxonomy": filtered_taxonomy, "summary": summary}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote filtered taxonomy to {output}")


if __name__ == "__main__":
    main()
