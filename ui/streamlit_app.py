"""Streamlit entry point for the BettingEdge dashboard."""
from __future__ import annotations

import os
import sys

# Ensure project root (repo root) is importable so "ui" and "engine" resolve.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# existing imports below
import subprocess
from datetime import datetime

import streamlit as st

import os
from pathlib import Path
import streamlit as st

# 1) Leggi il valore dai secrets e crea una cartella scrivibile su Cloud
tracking_uri = st.secrets.get("MLFLOW_TRACKING_URI", "file:/var/tmp/mlruns")
Path("/var/tmp/mlruns").mkdir(parents=True, exist_ok=True)  # /var/tmp è scrivibile su Cloud

# 2) Espone la variabile d'ambiente per eventuale codice che la legge da os.environ
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

# (facoltativo ma sicuro) Imposta anche via API MLflow se lo importi più sotto:
import mlflow
mlflow.set_tracking_uri(tracking_uri)


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
