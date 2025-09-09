from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .contracts import coerce_and_validate

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

_SEASON_RE = re.compile(r".*_(\d{2}_\d{2})\.csv")


def _season_from_filename(path: Path) -> str:
    m = _SEASON_RE.match(path.name)
    if not m:
        raise ValueError(f"Cannot infer season from filename: {path}")
    return m.group(1)


def unify(div: str, seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Load and unify raw CSV files for a division.

    Parameters
    ----------
    div:
        Division code, e.g. ``"I1"``.
    seasons:
        Iterable of season identifiers. If ``None`` or empty, all available
        seasons are used.
    """
    div_path = RAW_DIR / div
    files = sorted(div_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files found for division {div}")

    seasons_set = set(seasons or [])
    frames: list[pd.DataFrame] = []
    for f in files:
        season = _season_from_filename(f)
        if seasons_set and season not in seasons_set:
            continue
        df = pd.read_csv(f)
        rename_map = {
            "Date": "date",
            "Div": "div",
            "HomeTeam": "home",
            "AwayTeam": "away",
            "B365H": "odds_1",
            "B365D": "odds_x",
            "B365A": "odds_2",
            "FTHG": "ft_home_goals",
            "FTAG": "ft_away_goals",
        }
        df = df.rename(columns=rename_map)
        if "season" not in df.columns:
            df["season"] = season
        if "div" not in df.columns:
            df["div"] = div
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No files matched seasons {sorted(seasons_set)} for division {div}"
        )

    df = pd.concat(frames, ignore_index=True)
    return coerce_and_validate(df)
