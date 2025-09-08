"""Streamlit entry point for the BettingEdge dashboard."""
from __future__ import annotations

import subprocess
from datetime import datetime

import streamlit as st

from ui._state import DUCK_PATH, init_defaults
from ui._theme import inject_theme


st.set_page_config(page_title="bettingedge", layout="wide")
inject_theme()
init_defaults()


def _commit_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        )
    except Exception:
        return "unknown"


with st.sidebar:
    st.title("bettingedge")
    st.caption(f"{_commit_sha()} • {datetime.now().strftime('%Y-%m-%d')}")
    st.caption(f"DuckDB: {DUCK_PATH}")
    st.caption("Europe/Rome")
    st.divider()
    st.write("Select a page above to get started.")

st.write("Use the sidebar to navigate between pages.")
