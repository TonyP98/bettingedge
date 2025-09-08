"""Bookmaker odds fusion utilities.

This module will implement multi-book odds fusion using trimmed means,
robust statistics and outlier rejection techniques.
"""
from __future__ import annotations


def fuse_odds(odds: list[float]) -> float:
    """Placeholder implementation returning the average of provided odds."""
    if not odds:
        raise ValueError("odds list cannot be empty")
    return sum(odds) / len(odds)
