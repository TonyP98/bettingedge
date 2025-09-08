"""Evaluation schemas for DuckDB tables using Pandera."""
from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check

__all__ = ["dc_states_schema", "dc_preds_schema"]


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
