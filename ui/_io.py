"""I/O helpers for DuckDB access and artifact paths."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from ._state import DUCK_PATH


_cache_resource = getattr(st, "cache_resource", lambda **_: (lambda f: f))
_cache_data = getattr(st, "cache_data", lambda **_: (lambda f: f))


@_cache_resource()
def get_duck_conn() -> duckdb.DuckDBPyConnection:
    """Return a cached DuckDB connection."""
    path = Path(DUCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


@_cache_data(ttl=300)
def read_duck(query: str) -> pd.DataFrame:
    """Execute ``query`` against the DuckDB connection and return a DataFrame."""
    conn = get_duck_conn()
    return conn.execute(query).df()


def artifact_path(kind: str, run_id: str | int) -> Path:
    """Return path for a stored artifact HTML/CSV file."""
    return Path(f"data/processed/reports/{kind}_{run_id}.html")
