from __future__ import annotations

import pandas as pd


def implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied probabilities from decimal odds.

    Parameters
    ----------
    df:
        DataFrame containing columns ``odds_1``, ``odds_x`` and ``odds_2``.
    """
    df = df.copy()
    df["p1"] = 1 / df["odds_1"]
    df["px"] = 1 / df["odds_x"]
    df["p2"] = 1 / df["odds_2"]
    return df


def remove_vig(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the bookmaker's margin by normalising probabilities."""
    df = df.copy()
    total = df[["p1", "px", "p2"]].sum(axis=1)
    df[["p1", "px", "p2"]] = df[["p1", "px", "p2"]].div(total, axis=0)
    return df
