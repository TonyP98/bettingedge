"""Bet sizing utilities."""

from __future__ import annotations

import pandas as pd


def size_positions(
    signals: pd.DataFrame,
    mode: str = "fixed",
    fraction: float = 0.01,
    bankroll: float = 1.0,
) -> pd.DataFrame:
    """Attach stake sizes to a signals DataFrame.

    Parameters
    ----------
    signals:
        DataFrame produced by :func:`engine.signal.value.make_signals`.
    mode:
        ``"fixed"`` uses a flat fraction of ``bankroll`` for every bet.
        ``"kelly_f"`` applies a fractional Kelly criterion using ``fraction``
        as the multiplier of the optimal Kelly fraction.
    fraction:
        Fraction of bankroll to wager. For ``kelly_f`` it represents the
        fraction of the Kelly stake to use (e.g. ``0.25`` for 25% Kelly).
    bankroll:
        Current bankroll, defaults to 1 unit.
    """

    df = signals.copy()
    if df.empty:
        df["stake"] = []
        return df

    if mode == "fixed":
        df["stake"] = bankroll * fraction
    elif mode == "kelly_f":
        kelly_fraction = df["edge"] / (df["odds"] - 1)
        kelly_fraction = kelly_fraction.clip(lower=0)
        df["stake"] = bankroll * fraction * kelly_fraction
    else:
        raise ValueError(f"Unknown sizing mode: {mode}")

    return df


__all__ = ["size_positions"]

