"""Minimal what-if analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["simulate_delta"]


def simulate_delta(
    match_row: pd.Series,
    feature: str,
    v1: float,
    model,
    feature_names: Sequence[str],
    odds_col: str = "odds",
) -> dict:
    """Simulate the change in probability and edge for a single feature tweak.

    Parameters
    ----------
    match_row : pd.Series
        Row containing the feature values and odds.
    feature : str
        Name of the feature to modify.
    v1 : float
        New value to assign to ``feature``.
    model : classifier with ``predict_proba`` method.
    feature_names : sequence of str
        Names of the features expected by the model.
    odds_col : str, default ``"odds"``
        Column name holding the decimal odds used for edge computation.
    """
    X0 = match_row[feature_names].to_frame().T
    p0 = model.predict_proba(X0)[0, 1]
    odds = match_row[odds_col]
    edge0 = p0 * odds - 1

    X1 = X0.copy()
    X1[feature] = v1
    p1 = model.predict_proba(X1)[0, 1]
    edge1 = p1 * odds - 1

    return {
        "delta_p": p1 - p0,
        "delta_edge": edge1 - edge0,
        "p0": p0,
        "p1": p1,
        "edge0": edge0,
        "edge1": edge1,
    }
