from __future__ import annotations

from typing import Iterable
import pandas as pd

ESSENTIAL_COLUMNS: Iterable[str] = [
    "date",
    "div",
    "season",
    "home",
    "away",
    "odds_1",
    "odds_x",
    "odds_2",
    "ft_home_goals",
    "ft_away_goals",
]


_PLACEHOLDERS = {"n/d", "NA", "", " ", None}


def coerce_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce common placeholder values and enforce basic schema.

    Parameters
    ----------
    df: pd.DataFrame
        Raw dataframe possibly containing dirty values.

    Returns
    -------
    pd.DataFrame
        Clean dataframe with proper dtypes.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = [c for c in ESSENTIAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df.replace(list(_PLACEHOLDERS), pd.NA, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["season"].astype(str)
    for col in ["odds_1", "odds_x", "odds_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["ft_home_goals", "ft_away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["div", "home", "away"]:
        df[col] = df[col].astype(str)

    return df
