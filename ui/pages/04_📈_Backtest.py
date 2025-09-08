"""Streamlit backtest page with diagnostics tab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

try:  # pragma: no cover - optional
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

from engine.eval import backtester
from engine.eval.diagnostics import run_diagnostics
from engine.utils import mlflow_utils as mlf
from ui._state import get, set
from ui._widgets import equity_plot, metric_card

st.title("📈 Backtest")

backtest_tab, diag_tab = st.tabs(["Backtest", "Diagnostics"])

with backtest_tab:
    enable = st.checkbox("Enable Conformal Guard", value=False)
    if enable:
        alpha = st.slider("Alpha", 0.01, 0.5, 0.1)
        mondrian = st.multiselect(
            "Mondrian keys",
            ["Div", "Season", "Month", "OverroundBin", "RegimeId"],
            default=["Div"],
        )
        window = st.number_input(
            "Calibration window", min_value=1, max_value=50, value=10
        )
        edge_thr = st.number_input("Edge threshold", 0.0, 1.0, 0.02)
        kelly_cap = st.number_input("Kelly cap", 0.0, 1.0, 0.15)
        max_width = st.number_input("Max width", 0.0, 1.0, 0.35)
        st.write(
            f"Conformal guard enabled with α={alpha}, keys={mondrian}, window={window}, edge≥{edge_thr}"
        )
    else:
        st.write("Standard Kelly strategy without conformal guard.")

    if st.button("Run backtest"):
        try:
            cfg = OmegaConf.create(
                {
                    "conformal": {
                        "alpha": alpha if enable else 0.1,
                        "mondrian_keys": mondrian if enable else [],
                        "min_group_size": int(window) if enable else 10,
                        "width_cap": max_width if enable else 0.35,
                    },
                    "policy": {
                        "edge_thr": edge_thr if enable else 0.02,
                        "max_width": max_width if enable else 0.35,
                        "kelly_cap": kelly_cap if enable else 0.15,
                        "variance_penalty": 0.0,
                    },
                }
            )
            df = pd.DataFrame(
                {
                    "pH": [0.4, 0.35],
                    "pD": [0.3, 0.3],
                    "pA": [0.3, 0.35],
                    "oH": [2.1, 2.2],
                    "oD": [3.1, 3.0],
                    "oA": [3.2, 3.4],
                    "y": [0, 1],
                    "pnl": [0.1, -0.05],
                }
            )
            mlf.start_run("backtest")
            out = backtester.apply_conformal_guard(df, cfg)
            eq_df = pd.DataFrame(
                {"step": range(len(out)), "equity": out["pnl"].cumsum()}
            )
            eq_path = Path("data/processed/equity.csv")
            eq_path.parent.mkdir(parents=True, exist_ok=True)
            eq_df.to_csv(eq_path, index=False)
            mlf.log_artifact(str(eq_path))
            equity_plot(eq_df)
            diag_path = run_diagnostics()
            if isinstance(diag_path, str) and Path(diag_path).exists():
                mlf.log_artifact(diag_path)
                st.components.v1.html(
                    Path(diag_path).read_text(), height=300, scrolling=True
                )
            if mlflow and mlflow.active_run():
                run_id = mlflow.active_run().info.run_id
                set("last_run_id", run_id)
                st.success(f"MLflow run {run_id}")
                if st.button("Apri cartella artifact"):
                    st.write(mlflow.get_artifact_uri())
            mlf.end_run()
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

    st.header("Bandit (online)")
    algo = st.selectbox("Algoritmo", ["linucb", "thompson"], index=0)
    alpha_b = st.slider("Alpha", 0.1, 5.0, 1.0)
    epsilon_b = st.slider("Epsilon", 0.0, 0.5, 0.02)
    gamma = st.slider("Gamma CRRA", 1.0, 3.0, 2.0)
    decay = st.slider("Decay", 0.9, 1.0, 0.995)
    edge_grid = st.multiselect(
        "edge_thr grid",
        [0.01, 0.02, 0.03, 0.05, 0.08],
        default=[0.01, 0.02, 0.03, 0.05, 0.08],
    )
    kelly_grid = st.multiselect(
        "kelly_cap grid", [0.10, 0.15, 0.20, 0.25], default=[0.10, 0.15, 0.20, 0.25]
    )
    use_conf = st.checkbox("Usa Conformal Guard", value=True)

with diag_tab:
    st.header("Diagnostics")
    if st.button("Compute diagnostics"):
        try:
            diag_path = run_diagnostics()
            if isinstance(diag_path, str) and Path(diag_path).exists():
                st.components.v1.html(
                    Path(diag_path).read_text(), height=300, scrolling=True
                )
        except Exception as exc:
            st.error(f"Diagnostics failed: {exc}")
    if st.button("Open last report"):
        path = Path("data/processed/reports/diagnostics.html")
        if path.exists():
            st.components.v1.html(path.read_text(), height=300, scrolling=True)
        else:
            st.info("No diagnostics report found.")
    metric_card("ECE (ensemble)", "-")
    metric_card("Brier", "-")
    metric_card("KS-p (PIT)", "-")
    metric_card("# Regimes", "-")
