"""Streamlit backtest page with diagnostics tab."""
from __future__ import annotations

import pandas as pd
import streamlit as st

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
        window = st.number_input("Calibration window", min_value=1, max_value=50, value=10)
        edge_thr = st.number_input("Edge threshold", 0.0, 1.0, 0.02)
        kelly_cap = st.number_input("Kelly cap", 0.0, 1.0, 0.15)
        max_width = st.number_input("Max width", 0.0, 1.0, 0.35)
        st.write(
            f"Conformal guard enabled with α={alpha}, keys={mondrian}, window={window}, edge≥{edge_thr}"
        )
    else:
        st.write("Standard Kelly strategy without conformal guard.")

    if st.button("Run backtest"):
        st.info("Backtest logic not implemented; showing sample equity curve.")
        eq = pd.DataFrame({"step": [0, 1, 2, 3], "equity": [0, 1, 0.5, 1.5]})
        equity_plot(eq)

    st.header("Bandit (online)")
    algo = st.selectbox("Algoritmo", ["linucb", "thompson"], index=0)
    alpha_b = st.slider("Alpha", 0.1, 5.0, 1.0)
    epsilon_b = st.slider("Epsilon", 0.0, 0.5, 0.02)
    gamma = st.slider("Gamma CRRA", 1.0, 3.0, 2.0)
    decay = st.slider("Decay", 0.9, 1.0, 0.995)
    edge_grid = st.multiselect(
        "edge_thr grid", [0.01, 0.02, 0.03, 0.05, 0.08], default=[0.01, 0.02, 0.03, 0.05, 0.08]
    )
    kelly_grid = st.multiselect(
        "kelly_cap grid", [0.10, 0.15, 0.20, 0.25], default=[0.10, 0.15, 0.20, 0.25]
    )
    use_conf = st.checkbox("Usa Conformal Guard", value=True)

with diag_tab:
    st.header("Diagnostics")
    if st.button("Compute diagnostics"):
        st.info("Diagnostics computation triggered (placeholder).")
    if st.button("Open last report"):
        st.info("Opening last report not implemented.")
    metric_card("ECE (ensemble)", "-")
    metric_card("Brier", "-")
    metric_card("KS-p (PIT)", "-")
    metric_card("# Regimes", "-")
    st.markdown("[Scarica report HTML](#)")
