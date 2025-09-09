"""Streamlit entry point for the BettingEdge dashboard."""
from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import traceback

import pandas as pd
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient

# Ensure project root (repo root) is importable so "ui" and "engine" resolve.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# MLflow tracking URI resolution: env -> secrets -> fallback
tracking_uri = (
    os.environ.get("MLFLOW_TRACKING_URI")
    or st.secrets.get("MLFLOW_TRACKING_URI")
    or "file:/var/tmp/mlruns"
)
parsed_tracking = urlparse(tracking_uri)
if parsed_tracking.scheme == "file":
    Path(parsed_tracking.path).mkdir(parents=True, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
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


def render_mlflow_artifacts_viewer() -> None:
    client = MlflowClient()
    exp = client.get_experiment_by_name("backtest")
    if exp is None:
        st.info("Esegui Run backtest e riprova")
        return
    runs = client.search_runs(
        experiment_ids=exp.experiment_id,
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        st.info("Esegui Run backtest e riprova")
        return
    run_labels = [
        f"{r.info.run_id} | {r.info.status} | "
        f"{datetime.fromtimestamp(r.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S')}"
        for r in runs
    ]
    if "mlflow_run_index" not in st.session_state:
        st.session_state.mlflow_run_index = 0
    idx = st.selectbox(
        "Run",
        options=list(range(len(runs))),
        index=st.session_state.mlflow_run_index,
        format_func=lambda i: run_labels[i],
    )
    st.session_state.mlflow_run_index = idx
    run = runs[idx]
    artifact_uri = run.info.artifact_uri
    st.write("Artifact URI:", artifact_uri)
    parsed = urlparse(artifact_uri)
    if parsed.scheme != "file":
        st.warning("artifact store non locale: visualizzazione diretta non supportata")
        return
    artifact_dir = Path(parsed.path)
    st.write("Artifact path:", artifact_dir)
    equity_path = artifact_dir / "equity.csv"
    diagnostics_path = artifact_dir / "diagnostics.html"
    playbook_path = artifact_dir / "playbook.html"
    if equity_path.exists():
        try:
            df = pd.read_csv(equity_path)
            col = (
                "equity"
                if "equity" in df.columns
                else df.select_dtypes("number").columns.to_list()[0]
                if not df.select_dtypes("number").empty
                else None
            )
            if col:
                st.line_chart(df[col])
            else:
                st.write(df)
            st.download_button(
                "Download equity.csv", data=equity_path.read_bytes(), file_name="equity.csv"
            )
        except Exception:
            st.error(traceback.format_exc())
    else:
        st.warning("equity.csv non trovato")
    if diagnostics_path.exists():
        try:
            st.components.v1.html(
                diagnostics_path.read_text(), height=700, scrolling=True
            )
            st.download_button(
                "Download diagnostics.html",
                data=diagnostics_path.read_bytes(),
                file_name="diagnostics.html",
            )
        except Exception:
            st.error(traceback.format_exc())
    else:
        st.warning("diagnostics.html non trovato")
    if playbook_path.exists():
        try:
            st.components.v1.html(
                playbook_path.read_text(), height=700, scrolling=True
            )
            st.download_button(
                "Download playbook.html",
                data=playbook_path.read_bytes(),
                file_name="playbook.html",
            )
        except Exception:
            st.error(traceback.format_exc())
    else:
        st.warning("playbook.html non trovato")


with st.sidebar:
    st.title("bettingedge")
    st.caption(f"{_commit_sha()} • {datetime.now().strftime('%Y-%m-%d')}")
    st.caption(f"DuckDB: {DUCK_PATH}")
    st.caption("Europe/Rome")
    st.divider()
    st.write("Select a page above to get started.")

tab_home, tab_results = st.tabs(["Welcome", "Results / MLflow Artifacts"])
with tab_home:
    st.write("Use the sidebar to navigate between pages.")
with tab_results:
    render_mlflow_artifacts_viewer()
