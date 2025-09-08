"""Streamlit entry point."""
import streamlit as st
from pathlib import Path

from engine.data import ingest

st.title("BettingEdge Dashboard")

if st.button("Ingest data/raw"):
    tables = ingest.ingest("data/raw/l1_24_25.csv", commit=True)
    matches = tables["matches"]
    st.success(f"Ingested {len(matches)} matches")
    date_min = matches["event_time_local"].min()
    date_max = matches["event_time_local"].max()
    st.write(f"Date range: {date_min} - {date_max}")
    odds = tables["odds_1x2_pre"]
    pct_1x2 = odds.drop(columns=["MatchId"]).notna().any(axis=1).mean() * 100
    closing_pct = 0.0
    if "odds_1x2_close" in tables:
        close = tables["odds_1x2_close"].drop(columns=["MatchId"]).notna().any(axis=1)
        closing_pct = close.mean() * 100
    st.write(f"Rows with 1X2: {pct_1x2:.1f}% | closing 1X2: {closing_pct:.1f}%")
    st.write(f"OU columns present: {'ou_25_pre' in tables}")
    st.write(f"AH columns present: {'ah_pre' in tables}")

notes_path = Path("docs/specs/football-data-notes.md")
st.write(f"Notes: {notes_path}")
st.markdown("[Apri mappa chiavi](engine/data/specs/football_data_keys.yaml)")
