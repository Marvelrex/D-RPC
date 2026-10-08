#!/usr/bin/env python3
"""Shared helpers for proportional sampling logic."""

from __future__ import annotations

from typing import List, Sequence


def proportional_allocation(counts: Sequence[int], target: int) -> List[int]:
    """
    Allocate `target` items across buckets proportional to `counts`.

    Remainders are distributed deterministically by the largest fractional parts.
    """
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)

    raw = [c * target / total for c in counts]
    base = [int(x) for x in raw]
    remainder = target - sum(base)

    fracs = sorted(
        [(i, raw[i] - base[i]) for i in range(len(raw))],
        key=lambda x: x[1],
        reverse=True,
    )

    alloc = list(base)
    for idx, _ in fracs:
        if remainder <= 0:
            break
        alloc[idx] += 1
        remainder -= 1
    return alloc
