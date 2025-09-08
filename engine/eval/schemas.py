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
