from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Ensure that the directory for *path* exists.

    If *path* is a directory, it is created directly. If it is a file path,
    the parent directory is created. Returns the resolved directory path.
    """
    p = Path(path)
    directory = p if p.suffix == "" else p.parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame to *path*.

    The format is inferred from the file suffix (``.parquet`` or ``.csv``).
    """
    p = Path(path)
    ensure_dir(p)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported dataframe format: {p.suffix}")


def load_df(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from *path* based on its suffix."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported dataframe format: {p.suffix}")


def save_json(obj: object, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: str | Path) -> object:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)
