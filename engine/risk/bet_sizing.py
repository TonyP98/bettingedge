"""Bet sizing utilities implementing Kelly and risk adjustments."""
from __future__ import annotations


def size_bet(edge: float, bankroll: float) -> float:
    """Placeholder fractional Kelly criterion."""
    return max(0.0, edge) * bankroll
