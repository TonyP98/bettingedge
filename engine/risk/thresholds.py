"""Thresholding and bet sizing helpers for conformal strategies."""
from __future__ import annotations


def lower_edge_rule(p_low: float, odds: float, edge_thr: float) -> bool:
    """Return True if the lower-bound edge exceeds ``edge_thr``."""

    return p_low * odds - 1 >= edge_thr


def conformal_kelly(
    p_low: float,
    odds: float,
    kelly_cap: float,
    variance_penalty: float,
    width: float,
) -> float:
    """Kelly sizing using the lower-bound probability.

    Parameters
    ----------
    p_low: float
        Lower-bound probability for the selected outcome.
    odds: float
        Decimal odds for the outcome.
    kelly_cap: float
        Maximum fraction of bankroll to wager.
    variance_penalty: float
        Multiplier applied to penalise wide intervals.
    width: float
        Interval width for the selected class.
    """

    # Standard Kelly criterion
    edge = odds * p_low - 1
    if edge <= 0:
        return 0.0
    kelly = edge / (odds - 1)
    # Variance penalty
    kelly *= max(0.0, 1 - variance_penalty * width)
    # Cap
    return max(0.0, min(kelly, kelly_cap))
