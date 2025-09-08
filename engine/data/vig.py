"""Vig removal utilities."""
from __future__ import annotations


def remove_vig_multiplicative(odds: list[float]) -> list[float]:
    """Placeholder multiplicative vig removal."""
    total = sum(1 / o for o in odds)
    return [o * total for o in odds]


def remove_vig_shin(odds: list[float]) -> list[float]:
    """Placeholder Shin method vig removal."""
    # In real implementation we'd solve for the probability mass.
    return remove_vig_multiplicative(odds)
