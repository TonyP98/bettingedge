"""Centralized Streamlit session state keys and helpers."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


# -----------------------------------------------------------------------------
# Paths and session state defaults
# -----------------------------------------------------------------------------

# Resolve a single DATA_ROOT for the whole UI
_DATA_ROOT = (
    os.environ.get("DATA_ROOT")
    or st.secrets.get("DATA_ROOT", "data")
)

# Default path to DuckDB database used across the UI
DUCK_PATH = str(Path(_DATA_ROOT) / "processed" / "football.duckdb")


def init_defaults() -> None:
    """Ensure expected keys exist in ``st.session_state``."""
    st.session_state.setdefault("DATA_ROOT", _DATA_ROOT)
    st.session_state.setdefault("raw_paths", {})
    st.session_state.setdefault(
        "processed_paths",
        {"matches": DUCK_PATH, "odds_1x2_pre": DUCK_PATH, "market_probs_pre": DUCK_PATH},
    )
    st.session_state.setdefault("selected_bets", [])
    st.session_state.setdefault("last_run_id", None)
    st.session_state.setdefault("mlflow_run_id", None)
    st.session_state.setdefault("artifacts_root", None)


def get(key: str, default=None):
    """Retrieve a value from ``st.session_state`` with a default."""
    return st.session_state.get(key, default)


def set(key: str, value) -> None:
    """Store ``value`` under ``key`` in ``st.session_state``."""
    st.session_state[key] = value
