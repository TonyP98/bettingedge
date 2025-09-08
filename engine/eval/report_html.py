"""Utilities to generate HTML diagnostics reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import plotly.graph_objects as go


def generate_report_html(
    out_path: str,
    reliability_dfs: List[Any] | None = None,
    pit_u: Any | None = None,
    regime_kpi: Any | None = None,
    leakage_issues: Any | None = None,
) -> str:
    """Generate a minimal interactive HTML report and save it."""
    fig = go.Figure()
    fig.update_layout(title="Diagnostics Report")
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)


__all__ = ["generate_report_html"]
