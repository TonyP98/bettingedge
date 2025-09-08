"""Streamlit backtest page with conformal guard controls."""
import streamlit as st

st.title("📈 Backtest")

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
