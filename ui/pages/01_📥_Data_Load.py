"""Data loading and summary page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from engine.data import ingest
from engine.data.ingest import compute_market_probs, save_tables
from engine.data.contracts import MarketProbsSchema, validate_or_raise
from ui._io import read_duck

st.title("📥 Data Load")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded is not None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / uploaded.name
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved {dest}")

if st.button("Ingest data/raw"):
    with st.spinner("Ingesting..."):
        for path in Path("data/raw").glob("*.csv"):
            ingest.ingest(str(path), commit=True)
    st.success("Ingest completed")

st.subheader("Summary")
try:
    matches = read_duck("SELECT * FROM matches")
    if not matches.empty:
        date_min = matches["event_time_local"].min()
        date_max = matches["event_time_local"].max()
        st.write(f"Matches: {len(matches)} | Date range: {date_min} - {date_max}")
    else:
        st.info("No matches present.")
except Exception:
    st.info("Matches table not available.")

st.subheader("Specs")
st.markdown("[Notes](docs/specs/football-data-notes.md)")
st.markdown("[Key map](engine/data/specs/football_data_keys.yaml)")


if st.button("Rebuild Market Probs"):
    try:
        matches = read_duck("SELECT * FROM matches")
        odds_pre = read_duck("SELECT * FROM odds_1x2_pre")
        try:
            odds_close = read_duck("SELECT * FROM odds_1x2_close")
        except Exception:
            odds_close = None
        probs = compute_market_probs(matches, odds_pre, odds_close)
        try:
            if "market_probs_pre" in probs:
                validate_or_raise(
                    probs["market_probs_pre"][["pH_mul", "pD_mul", "pA_mul"]].rename(
                        columns={"pH_mul": "pH", "pD_mul": "pD", "pA_mul": "pA"}
                    ),
                    MarketProbsSchema,
                    "market_probs_pre",
                )
            if "market_probs_close" in probs:
                validate_or_raise(
                    probs["market_probs_close"][["pH_mul", "pD_mul", "pA_mul"]].rename(
                        columns={"pH_mul": "pH", "pD_mul": "pD", "pA_mul": "pA"}
                    ),
                    MarketProbsSchema,
                    "market_probs_close",
                )
        except Exception as e:
            st.error(f"Validazione fallita: {e}")
            raise
        save_tables(probs)
        st.success("Market probabilities ricostruite e tabelle salvate.")
        if isinstance(probs, dict) and probs:
            name, df = next(iter(probs.items()))
            st.write(f"Preview of {name}")
            st.dataframe(df.head(50))
    except Exception as e:
        import traceback
        st.error(f"Errore durante il rebuild delle market probs: {e}")
        st.code("".join(traceback.format_exc()))
