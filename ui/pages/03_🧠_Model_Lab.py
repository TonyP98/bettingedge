"""Streamlit page for model experiments."""
import pandas as pd
import streamlit as st

from engine.models.dc_state_space import DCStateSpace

st.title("🧠 Model Lab")
window = st.selectbox("Rolling window", list(range(4, 13)), index=4)
max_goals = st.number_input("Max goals", min_value=1, max_value=10, value=6)
max_em_iter = st.number_input("Max EM iterations", min_value=1, max_value=20, value=8)

if st.button("Fit DC (state-space)"):
    # Placeholder dataset for demonstration
    df = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [1],
            "away_goals": [0],
        }
    )
    model = DCStateSpace({"max_goals": int(max_goals), "max_em_iter": int(max_em_iter)})
    model.fit(df)
    preds = model.predict_df(df)
    st.write("Predictions:")
    st.dataframe(preds)
    st.write("States snapshot:")
    st.write({"atk": model.atk, "def": model.def_})
