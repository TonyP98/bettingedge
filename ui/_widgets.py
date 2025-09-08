"""Reusable Streamlit widgets for the dashboard."""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd


def metric_card(title: str, value, help: str | None = None) -> None:
    """Display a metric card with optional help tooltip."""
    st.metric(title, value, help=help)


def equity_plot(df: pd.DataFrame) -> None:
    """Render a simple equity curve using Plotly."""
    if df.empty:
        st.info("No equity data available.")
        return
    fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Equity")
    st.plotly_chart(fig, use_container_width=True)


def reliability_plot(df: pd.DataFrame) -> None:
    """Render a reliability curve (calibration)."""
    if df.empty:
        st.info("No reliability data available.")
        return
    fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Reliability")
    st.plotly_chart(fig, use_container_width=True)
