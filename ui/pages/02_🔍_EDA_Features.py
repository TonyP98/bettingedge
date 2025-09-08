"""EDA and feature preview page."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from ui._io import read_duck

st.title("🔍 EDA & Features")

try:
    df = read_duck("SELECT FTR, FTHG, FTAG, overround FROM matches")
    if df.empty:
        st.info("No match data available.")
    else:
        st.plotly_chart(px.histogram(df, x="FTR"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="FTHG", nbins=6), use_container_width=True)
        st.plotly_chart(px.line(df, y="overround"), use_container_width=True)
except Exception:
    st.info("Matches table not available.")

if st.button("Build Features"):
    try:
        from engine.features import builder

        builder.build_features()
        st.success("Features built")
    except Exception as exc:
        st.error(f"Feature builder not available: {exc}")

try:
    feat = read_duck("SELECT * FROM features LIMIT 200")
    st.dataframe(feat)
except Exception:
    st.info("Features table not available.")
