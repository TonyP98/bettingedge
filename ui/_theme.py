"""Minimal theme utilities for Streamlit."""
from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    """Inject light CSS tweaks for metric cards and fonts."""
    st.markdown(
        """
        <style>
        .metric {background-color: rgba(0,0,0,0.03); padding: 0.5rem; border-radius: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
