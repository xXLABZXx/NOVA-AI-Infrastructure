"""Evaluation metrics: normalized exact match and accuracy.

Normalization (strip, collapse whitespace, lowercase) makes exact match
robust to trivial formatting differences.
"""

from __future__ import annotations

import re


def normalize(text) -> str:
    """Normalize a prediction/expected string for comparison."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def exact_match(pred, expected) -> bool:
    """True when ``pred`` and ``expected`` match after normalization."""
    return normalize(pred) == normalize(expected)


def accuracy(pairs) -> float:
    """Fraction of ``(pred, expected)`` pairs that exactly match.

    Returns 0.0 for an empty input.
    """
    pairs = list(pairs)
    if not pairs:
        return 0.0
    correct = sum(1 for pred, expected in pairs if exact_match(pred, expected))
    return correct / len(pairs)


def score_pair(pred, expected) -> dict:
    """Per-pair score detail: normalized strings plus the match result."""
    return {
        "pred": normalize(pred),
        "expected": normalize(expected),
        "exact_match": exact_match(pred, expected),
    }
