from __future__ import annotations

import os
from pathlib import Path


def get_data_root() -> str:
    """Return base data directory from env, secrets or default."""
    import streamlit as st
    return os.environ.get("DATA_ROOT") or getattr(st, "secrets", {}).get("DATA_ROOT", "data")


def get_duck_path() -> str:
    """Compute path to the shared DuckDB database."""
    return str(Path(get_data_root()) / "processed" / "football.duckdb")


def init_defaults() -> None:
    """Ensure expected keys exist in ``st.session_state``."""
    import streamlit as st

    data_root = get_data_root()
    duck_path = get_duck_path()
    st.session_state.setdefault("DATA_ROOT", data_root)
    st.session_state.setdefault("raw_paths", {})
    processed = st.session_state.setdefault("processed_paths", {})
    processed.setdefault("matches", duck_path)
    processed.setdefault("odds_1x2_pre", duck_path)
    processed.setdefault("market_probs_pre", duck_path)
    st.session_state.setdefault("selected_bets", [])
    st.session_state.setdefault("last_run_id", None)
    st.session_state.setdefault("mlflow_run_id", None)
    st.session_state.setdefault("artifacts_root", None)


def get(key: str, default=None):
    """Retrieve a value from ``st.session_state`` with a default."""
    import streamlit as st
    return st.session_state.get(key, default)


def set(key: str, value) -> None:
    """Store ``value`` under ``key`` in ``st.session_state``."""
    import streamlit as st
    st.session_state[key] = value


# Backwards compatibility for modules expecting a constant
DUCK_PATH = get_duck_path()
