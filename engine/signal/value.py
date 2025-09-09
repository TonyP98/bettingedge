"""Computation of value-based betting signals.

This module provides utilities to compare model probabilities against the
market and extract positive expected value opportunities.
"""

from __future__ import annotations

import pandas as pd

_SEL_MAP = {
    "1": ("p1", "odds_1"),
    "x": ("px", "odds_x"),
    "2": ("p2", "odds_2"),
}


def compute_value(probs: pd.Series, odds: pd.Series) -> pd.Series:
    """Return the expected value for each outcome.

    Parameters
    ----------
    probs:
        Model probabilities for the selection.
    odds:
        Corresponding decimal odds from the market.
    """

    return probs * odds - 1


def make_signals(
    df_probs: pd.DataFrame,
    ev_min: float = 0.0,
    odds_min: float | None = None,
    odds_max: float | None = None,
    max_per_day: int | None = None,
) -> pd.DataFrame:
    """Create a long-form DataFrame of value signals.

    Parameters
    ----------
    df_probs:
        DataFrame with columns ``date``, ``home``, ``away``, ``result`` and
        probability/odds columns for each selection.
    ev_min:
        Minimum expected value threshold. Only selections with ``edge`` greater
        than or equal to ``ev_min`` are retained.
    odds_min / odds_max:
        Optional bounds on allowed odds.
    max_per_day:
        If given, limit the number of picks per date to the selections with the
        highest edges.
    """

    df = df_probs.copy()
    df["date"] = pd.to_datetime(df["date"])

    records: list[pd.DataFrame] = []
    for sel, (p_col, o_col) in _SEL_MAP.items():
        p_model = df[p_col]
        odds = df[o_col]
        p_market = 1 / odds
        edge = compute_value(p_model, odds)

        mask = edge >= ev_min
        if odds_min is not None:
            mask &= odds >= odds_min
        if odds_max is not None:
            mask &= odds <= odds_max

        subset = df.loc[mask, ["date", "home", "away", "result"]].copy()
        subset["selection"] = sel
        subset["odds"] = odds[mask]
        subset["p_model"] = p_model[mask]
        subset["p_market"] = p_market[mask]
        subset["edge"] = edge[mask]
        records.append(subset)

    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "match_id",
                "home",
                "away",
                "selection",
                "odds",
                "p_model",
                "p_market",
                "edge",
                "result",
            ]
        )

    signals = pd.concat(records, ignore_index=True)
    signals.sort_values("date", inplace=True)

    # Generate a simple match identifier based on ordering of matches
    signals.insert(1, "match_id", signals.groupby(["date", "home", "away"]).ngroup())

    if max_per_day is not None:
        signals = (
            signals.groupby("date", group_keys=False)
            .apply(lambda x: x.nlargest(max_per_day, "edge"))
            .reset_index(drop=True)
        )

    return signals[
        [
            "date",
            "match_id",
            "home",
            "away",
            "selection",
            "odds",
            "p_model",
            "p_market",
            "edge",
            "result",
        ]
    ]

