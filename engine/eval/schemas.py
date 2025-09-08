"""Evaluation schemas for DuckDB tables using Pandera."""
from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check

__all__ = [
    "dc_states_schema",
    "dc_preds_schema",
    "ensemble_preds_schema",
    "conformal_calibs_schema",
    "bet_logs_conformal_schema",
    "bandit_logs_schema",
    "diagnostics_summary_schema",
    "diagnostics_reliability_schema",
    "diagnostics_pit_schema",
    "diagnostics_regimes_schema",
]


dc_states_schema = DataFrameSchema(
    {
        "team_id": Column(pa.String),
        "t": Column(pa.Int64),
        "atk": Column(pa.Float64),
        "def": Column(pa.Float64),
        "atk_var": Column(pa.Float64),
        "def_var": Column(pa.Float64),
        "rho": Column(pa.Float64),
        "home_adv": Column(pa.Float64),
        "season": Column(pa.String),
        "updated_at": Column(pa.DateTime),
    },
    coerce=True,
)


dc_preds_schema = DataFrameSchema(
    {
        "match_id": Column(pa.String),
        "t": Column(pa.Int64),
        "ph": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "pd": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "pa": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "pou25": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "lambda_h": Column(pa.Float64),
        "lambda_a": Column(pa.Float64),
        "rho": Column(pa.Float64),
        "method": Column(pa.String),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)


ensemble_preds_schema = DataFrameSchema(
    {
        "match_id": Column(pa.String),
        "t": Column(pa.Int64),
        "ph": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "pd": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "pa": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "method": Column(pa.String),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)


conformal_calibs_schema = DataFrameSchema(
    {
        "mondrian_key": Column(pa.String),
        "fold": Column(pa.Int64),
        "cls": Column(pa.String),
        "alpha": Column(pa.Float64),
        "coverage": Column(pa.Float64),
        "width_avg": Column(pa.Float64),
        "fitted_at": Column(pa.DateTime),
    },
    coerce=True,
)


bet_logs_conformal_schema = DataFrameSchema(
    {
        "match_id": Column(pa.String),
        "t": Column(pa.Int64),
        "sel_cls": Column(pa.String),
        "odds": Column(pa.Float64),
        "p_low": Column(pa.Float64),
        "p_hat": Column(pa.Float64),
        "edge_low": Column(pa.Float64),
        "stake": Column(pa.Float64),
        "pnl": Column(pa.Float64),
        "params": Column(pa.String),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)

bandit_logs_schema = DataFrameSchema(
    {
        "t": Column(pa.Int64),
        "match_id": Column(pa.String),
        "action_edge_thr": Column(pa.Float64),
        "action_kcap": Column(pa.Float64),
        "context": Column(pa.Object),
        "reward": Column(pa.Float64),
        "stake": Column(pa.Float64),
        "pnl": Column(pa.Float64),
        "bankroll": Column(pa.Float64),
        "algo": Column(pa.String),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)


diagnostics_summary_schema = DataFrameSchema(
    {
        "fold": Column(pa.Int64),
        "metric": Column(pa.String),
        "value": Column(pa.Float64),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)


diagnostics_reliability_schema = DataFrameSchema(
    {
        "fold": Column(pa.Int64),
        "bin": Column(pa.Int64),
        "p_hat": Column(pa.Float64),
        "freq": Column(pa.Float64),
        "model": Column(pa.String),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)


diagnostics_pit_schema = DataFrameSchema(
    {
        "fold": Column(pa.Int64),
        "model": Column(pa.String),
        "u": Column(pa.Float64, Check.in_range(0, 1, inclusive="both")),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)


diagnostics_regimes_schema = DataFrameSchema(
    {
        "fold": Column(pa.Int64),
        "regime_id": Column(pa.Int64),
        "n": Column(pa.Int64),
        "kpi_json": Column(pa.Object),
        "created_at": Column(pa.DateTime),
    },
    coerce=True,
)
