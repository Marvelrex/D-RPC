#!/usr/bin/env python3
"""
Build a three-level taxonomy from structured rationale outputs.

Reads a JSONL results file, filters to gold-matching entries, clusters intents
per category using SentenceTransformer embeddings, PCA (5D), and DBSCAN, then
prints Category -> canonical Intent -> reasoning_paths. Optionally deduplicates
reasoning paths with word2vec/token-overlap similarity after clustering.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, HELPER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
except ImportError as exc:
    raise SystemExit(
        "This script requires 'sentence-transformers' and 'scikit-learn'. "
        "Install them with: pip install sentence-transformers scikit-learn"
    ) from exc

from filter_taxonomy_similar import (
    SimilarityComputer,
    filter_taxonomy,
    load_word2vec,
    recompute_summary,
)


def load_entries(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def trim_words(text: str, max_words: int = 4) -> str:
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def pick_canonical_intent(intents: List[str]) -> str:
    if not intents:
        return "unknown"
    shortest = min(intents, key=lambda s: (len(s.split()), len(s)))
    return trim_words(shortest, max_words=4)


def cluster_intents(
    intents: List[str],
    model: SentenceTransformer,
    eps: float = 0.25,
    min_samples: int = 2,
) -> Dict[str, List[str]]:
    """Return mapping canonical_intent -> member intents."""
    if not intents:
        return {}

    embeddings = model.encode(intents, convert_to_numpy=True, show_progress_bar=False)
    n_components = min(5, embeddings.shape[0], embeddings.shape[1])
    if n_components < 1:
        return {trim_words(intents[0], 4): [intents[0]]}
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(reduced)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for intent, label in zip(intents, labels):
        clusters[label].append(intent)

    canonical_map: Dict[str, List[str]] = {}
    for label, members in clusters.items():
        if label == -1:
            for m in members:
                canonical_map[trim_words(m, 4)] = [m]
        else:
            canonical = pick_canonical_intent(members)
            canonical_map[canonical] = members
    return canonical_map


def build_taxonomy(
    entries: Iterable[dict],
    eps: float = 0.25,
    min_samples: int = 2,
) -> Dict[str, Dict[str, List[List[str]]]]:
    """category -> canonical_intent -> list of reasoning_paths."""
    data: Dict[str, Dict[str, List[List[str]]]] = defaultdict(lambda: defaultdict(list))

    all_categories = set()
    raw_intents = set()
    all_reasoning_paths = set()
    canonical_intents = set()

    per_category_intents: Dict[str, List[str]] = defaultdict(list)
    per_category_paths: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)

    def _coerce_dict(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return {}

    for entry in entries:
        if not entry.get("gold_matches_answer"):
            continue
        route = _coerce_dict(entry.get("route"))
        if not route:
            payload = _coerce_dict(entry.get("response_payload"))
            if payload:
                route = _coerce_dict(payload.get("route"))
        if not route:
            route = {}

        category = str(route.get("category") or "unknown").strip()
        intent_list = route.get("intent") or []
        reasoning_path = route.get("reasoning_path") or []
        if not category or not intent_list or not reasoning_path:
            continue
        all_categories.add(category)
        all_reasoning_paths.add(tuple(reasoning_path))
        for intent in intent_list:
            intent = str(intent).strip()
            if not intent:
                continue
            raw_intents.add(intent)
            per_category_intents[category].append(intent)
            per_category_paths[(category, intent)].append(reasoning_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    for category, intent_list in per_category_intents.items():
        canonical_map = cluster_intents(intent_list, model=model, eps=eps, min_samples=min_samples)
        intent_to_canonical = {}
        for canonical, members in canonical_map.items():
            for m in members:
                intent_to_canonical[m] = canonical
            canonical_intents.add(canonical)

        for intent, paths in per_category_paths.items():
            cat, intent_text = intent
            if cat != category:
                continue
            canonical = intent_to_canonical.get(intent_text, trim_words(intent_text, 4))
            data[category][canonical].extend(paths)

    for cat, intents in data.items():
        for intent, paths in intents.items():
            seen = set()
            unique = []
            for p in paths:
                tup = tuple(p)
                if tup in seen:
                    continue
                seen.add(tup)
                unique.append(p)
            intents[intent] = unique

    stats = {
        "unique_categories": len(all_categories),
        "unique_intents": len(raw_intents),
        "unique_reasoning_paths": len(all_reasoning_paths),
        "unique_canonical_intents": len(canonical_intents),
        "raw_intents_before_clustering": len(raw_intents),
    }
    return data, stats


def print_taxonomy(taxonomy: Dict[str, Dict[str, List[List[str]]]]) -> None:
    for category in sorted(taxonomy):
        print(f'Category: "{category}"')
        intents = taxonomy[category]
        for intent in sorted(intents):
            print(f' ├── Intent: "{intent}"')
            for path in intents[intent]:
                print(f" │     ├── ReasoningPath: {path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce a reasoning taxonomy from results.jsonl")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("data/structure_rationale/results.jsonl"),
        help="Path to results JSONL file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/reasoning_bank/taxonomy.json"),
        help="Where to write the taxonomy JSON (default: data/reasoning_bank/taxonomy.json).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing output to disk; just print to stdout.",
    )
    parser.add_argument(
        "--filter-similar",
        action="store_true",
        help="After clustering, remove highly similar reasoning paths using word2vec/token-overlap similarity.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for filtering redundant reasoning paths (used when --filter-similar).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.25,
        help="DBSCAN eps for intent clustering (default: 0.25).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples for intent clustering (default: 2).",
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

    entries = list(load_entries(args.results_path))
    taxonomy, stats = build_taxonomy(entries, eps=args.eps, min_samples=args.min_samples)

    if args.filter_similar:
        model = load_word2vec(args.w2v_model, args.w2v_binary)
        sim = SimilarityComputer(model)
        filtered_taxonomy, _, _ = filter_taxonomy(taxonomy, sim, args.sim_threshold)
        taxonomy = filtered_taxonomy
        stats = recompute_summary(
            taxonomy, raw_intents_before_clustering=stats.get("raw_intents_before_clustering")
        )

    print_taxonomy(taxonomy)
    print("Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if not args.no_write:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "taxonomy": taxonomy,
            "summary": stats,
        }
        with args.output_path.open("w", encoding="utf-8") as fh:
            json.dump(out_data, fh, ensure_ascii=False, indent=2)
        print(f"Wrote taxonomy to {args.output_path}")


if __name__ == "__main__":
    main()
