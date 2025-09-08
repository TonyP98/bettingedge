"""Bookmaker odds fusion utilities.

This module implements a simple trimmed mean fusion for 1X2 odds across
multiple bookmakers. It also reports the number of contributing books and
basic spread information (difference between max and min odds).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import statistics
import pandas as pd

from .contracts import Odds1x2PreSchema, validate_or_raise


def _trimmed_mean(values: List[float]) -> float:
    vals = sorted(values)
    if len(vals) > 2:
        vals = vals[1:-1]
    return statistics.mean(vals)


def fuse_1x2(
    odds_by_book: Iterable[Tuple[float, float, float]],
) -> Tuple[float, float, float, int, Tuple[float, float, float]]:
    """Fuse odds from multiple bookmakers using a trimmed mean.

    Parameters
    ----------
    odds_by_book : iterable of (H, D, A)
        Odds from each bookmaker.

    Returns
    -------
    tuple
        (fusedH, fusedD, fusedA, n_books, (spreadH, spreadD, spreadA))
    """
    odds_list = list(odds_by_book)
    if not odds_list:
        raise ValueError("odds_by_book cannot be empty")

    n_books = len(odds_list)
    home_vals = [o[0] for o in odds_list]
    draw_vals = [o[1] for o in odds_list]
    away_vals = [o[2] for o in odds_list]

    fusedH = _trimmed_mean(home_vals)
    fusedD = _trimmed_mean(draw_vals)
    fusedA = _trimmed_mean(away_vals)

    overround = sum(1.0 / x for x in (fusedH, fusedD, fusedA))
    validate_or_raise(
        pd.DataFrame(
            {
                "H": [fusedH],
                "D": [fusedD],
                "A": [fusedA],
                "overround_pre": [overround],
            }
        ),
        Odds1x2PreSchema,
        "fuse_1x2",
    )

    spreadH = max(home_vals) - min(home_vals)
    spreadD = max(draw_vals) - min(draw_vals)
    spreadA = max(away_vals) - min(away_vals)

    return fusedH, fusedD, fusedA, n_books, (spreadH, spreadD, spreadA)
