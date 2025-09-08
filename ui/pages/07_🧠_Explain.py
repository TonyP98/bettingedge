"""Explainability dashboard placeholder."""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explain", layout="wide")

st.title("🧠 Explain")

st.header("Global Importance")
st.info("Global SHAP values will appear here.")

st.header("Local Cases")
st.info("Select a match to view local SHAP waterfall plot.")

st.header("Rule Playbook")
st.info("Discovered rules and their KPIs will be listed here.")

st.header("What-If Analysis")
st.info("Interactively tweak features to see edge changes.")
