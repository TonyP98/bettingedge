"""Centralized Streamlit session state keys and helpers."""
from __future__ import annotations

import streamlit as st

# Default path to DuckDB database used across the UI
DUCK_PATH = "data/processed/football.duckdb"

# Common session keys
def init_defaults() -> None:
    """Ensure expected keys exist in ``st.session_state``."""
    if "selected_bets" not in st.session_state:
        st.session_state["selected_bets"] = []
    if "last_run_id" not in st.session_state:
        st.session_state["last_run_id"] = None


def get(key: str, default=None):
    """Retrieve a value from ``st.session_state`` with a default."""
    return st.session_state.get(key, default)


def set(key: str, value) -> None:
    """Store ``value`` under ``key`` in ``st.session_state``."""
    st.session_state[key] = value
