"""Bet sizing simulation page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui._state import get

st.title("🎯 Bet Sizing Simulator")

bets = get("selected_bets", [])
if not bets:
    st.info("No bets selected. Use Edge Finder to choose some.")
else:
    df = pd.DataFrame(bets)
    st.dataframe(df)
    stake_policy = st.selectbox("Stake policy", ["flat", "kelly"], index=0)
    cap = st.number_input("Cap", 0.0, 1.0, 0.1)
    if st.button("Simulate"):
        probs = df["p_blend"].values
        odds = df["odds"].values
        rng = np.random.default_rng(0)
        res = []
        for _ in range(1000):
            outcome = rng.binomial(1, probs)
            pnl = (outcome * odds - 1).sum()
            res.append(pnl)
        st.write(f"PnL mean: {np.mean(res):.3f}")
        st.download_button("Download CSV", df.to_csv(index=False), "sizing.csv")
