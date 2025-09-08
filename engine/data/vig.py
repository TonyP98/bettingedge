"""Vig removal utilities."""
from __future__ import annotations

from typing import Tuple
import math


def remove_vig_multiplicative(oddsH: float, oddsD: float, oddsA: float) -> Tuple[float, float, float]:
    """Remove vig using a simple multiplicative model.

    Parameters
    ----------
    oddsH, oddsD, oddsA : float
        The home, draw and away odds.

    Returns
    -------
    tuple of float
        Probabilities normalised to sum to one.
    """
    invs = [1.0 / oddsH, 1.0 / oddsD, 1.0 / oddsA]
    total = sum(invs)
    return tuple(x / total for x in invs)


def _shin_transform(q: float, z: float) -> float:
    return (math.sqrt(z * z + 4 * (1 - z) * q) - z) / (2 * (1 - z))


def remove_vig_shin(oddsH: float, oddsD: float, oddsA: float) -> Tuple[float, float, float]:
    """Remove vig using the Shin method.

    This solves for the parameter ``z`` that represents the proportion of
    insider trading in the market and uses it to derive fair probabilities.
    The implementation uses a simple bisection search which is sufficient for
    the small systems encountered in unit tests.
    """
    q = [1.0 / oddsH, 1.0 / oddsD, 1.0 / oddsA]

    def objective(z: float) -> float:
        return sum(_shin_transform(qi, z) for qi in q) - 1.0

    lo, hi = 0.0, 0.999999
    for _ in range(100):
        mid = (lo + hi) / 2
        val = objective(mid)
        if abs(val) < 1e-9:
            break
        if val > 0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    probs = [_shin_transform(qi, z) for qi in q]
    total = sum(probs)
    return tuple(p / total for p in probs)

