#!/usr/bin/env python3
"""Shared tokenization helpers for taxonomy label strings."""

from __future__ import annotations

import re
from typing import List

TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")


def split_tokens(label: str) -> List[str]:
    """Lowercase token list from a label such as CamelCase or snake_case."""
    return [tok.lower() for tok in TOKEN_PATTERN.findall(label) if tok]
