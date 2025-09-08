"""Utilities for preparing rolling train/test splits."""
from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import pandas as pd

__all__ = ["rolling_window_split", "walk_forward_split"]


def rolling_window_split(df: pd.DataFrame, window: int, step: int = 1) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate rolling window train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe sorted by time.
    window : int
        Number of observations to include in each training window.
    step : int, default 1
        Step size between windows.

    Yields
    ------
    tuple(pd.DataFrame, pd.DataFrame)
        Train and test dataframes for each split.
    """
    n = len(df)
    for start in range(0, n - window, step):
        train = df.iloc[start : start + window]
        test = df.iloc[start + window : start + window + step]
        if len(test) == 0:
            break
        yield train, test


def walk_forward_split(df: pd.DataFrame, t_col: str = "t") -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield splits where training data is strictly prior to the test period."""
    ts = sorted(df[t_col].unique())
    for t in ts[1:]:
        train = df[df[t_col] < t]
        test = df[df[t_col] == t]
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test
