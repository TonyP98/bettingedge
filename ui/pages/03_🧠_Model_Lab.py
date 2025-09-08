"""Streamlit page for model experiments."""
import pandas as pd
import streamlit as st

from engine.models.dc_state_space import DCStateSpace
from engine.models.bivar_poisson_corr import BivarPoisson
from engine.models.meta_learner import MetaEnsemble
from engine.eval.feature_assembly import assemble_ensemble_features

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

st.header("Ensemble")
if st.button("Fit Bivariate Poisson (rolling)"):
    df = pd.DataFrame(
        {
            "atk_home": [1.0],
            "def_home": [0.5],
            "atk_away": [0.8],
            "def_away": [0.6],
            "home_adv": [0.1],
            "z1": [0.2],
            "z2": [-0.1],
            "z3": [1.0],
            "home_goals": [1],
            "away_goals": [0],
        }
    )
    bp = BivarPoisson().fit(df, {"max_goals": int(max_goals)})
    st.dataframe(bp.predict_df(df))

if st.button("Build Ensemble (stacking + calibration)"):
    df = pd.DataFrame(
        {
            "pH_DC": [0.4],
            "pD_DC": [0.3],
            "pA_DC": [0.3],
            "pOU_DC": [0.5],
            "pH_BP": [0.45],
            "pD_BP": [0.25],
            "pA_BP": [0.30],
            "pOU_BP": [0.52],
            "pH_MKT_pre": [0.42],
            "pD_MKT_pre": [0.28],
            "pA_MKT_pre": [0.30],
            "form_diff": [0.1],
            "micro_overround": [0.02],
            "y_wdl": [0],
        }
    )
    X, y = assemble_ensemble_features(df)
    meta = MetaEnsemble()
    meta.fit(X, y)
    proba = meta.predict_proba(X)
    st.dataframe(pd.DataFrame(proba, columns=["pH_blend", "pD_blend", "pA_blend"]))
