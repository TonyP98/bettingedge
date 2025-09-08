"""Edge finder page allowing selection of candidate bets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui._state import get, set

st.title("🕵️ Edge Finder")

# Placeholder dataset
cand = pd.DataFrame(
    {
        "match": ["A-B", "C-D"],
        "p_blend": [0.45, 0.55],
        "odds": [2.1, 1.8],
        "edge_low": [0.05, 0.01],
    }
)

selection = st.dataframe(cand, use_container_width=True)

if st.button("Invia primo a Sizing"):
    bets = get("selected_bets", [])
    bets.append(cand.iloc[0].to_dict())
    set("selected_bets", bets)
    st.success("Bet added to sizing simulator")
