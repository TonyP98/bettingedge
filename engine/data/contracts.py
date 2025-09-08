"""Data contracts defined using Pandera schemas."""

from __future__ import annotations

from typing import Type

import pandas as pd

try:  # pragma: no cover - optional dependency
    import pandera as pa
except Exception:  # pragma: no cover - fallback
    pa = None  # type: ignore

if pa:
    MatchesSchema = pa.DataFrameSchema(
        {
            "Div": pa.Column(str),
            "HomeTeam": pa.Column(str),
            "AwayTeam": pa.Column(str),
            "FTHG": pa.Column(int, checks=pa.Check.ge(0)),
            "FTAG": pa.Column(int, checks=pa.Check.ge(0)),
            "FTR": pa.Column(str, checks=pa.Check.isin(["H", "D", "A"])),
        }
    )

    Odds1x2PreSchema = pa.DataFrameSchema(
        {
            "H": pa.Column(float, checks=[pa.Check.ge(1.01), pa.Check.le(200)]),
            "D": pa.Column(float, checks=[pa.Check.ge(1.01), pa.Check.le(200)]),
            "A": pa.Column(float, checks=[pa.Check.ge(1.01), pa.Check.le(200)]),
            "overround_pre": pa.Column(float, checks=pa.Check.le(1.25)),
        }
    )

    MarketProbsSchema = pa.DataFrameSchema(
        {
            "pH": pa.Column(
                float,
                checks=pa.Check(lambda s: s.isna() | ((s >= 0) & (s <= 1))),
                nullable=True,
                required=False,
            ),
            "pD": pa.Column(
                float,
                checks=pa.Check(lambda s: s.isna() | ((s >= 0) & (s <= 1))),
                nullable=True,
                required=False,
            ),
            "pA": pa.Column(
                float,
                checks=pa.Check(lambda s: s.isna() | ((s >= 0) & (s <= 1))),
                nullable=True,
                required=False,
            ),
        },
        checks=pa.Check(
            lambda df: (
                df[["pH", "pD", "pA"]]
                .dropna()
                .apply(lambda r: abs(r.sum() - 1) <= 1e-6, axis=1)
                .all()
            )
        ),
    )
else:  # pragma: no cover - placeholders when pandera unavailable
    MatchesSchema = object  # type: ignore
    Odds1x2PreSchema = object  # type: ignore
    MarketProbsSchema = object  # type: ignore


def validate_or_raise(df, schema: Type, name: str):
    """Validate ``df`` against ``schema`` if Pandera is available."""
    if pa is None:
        return df
    try:
        return schema.validate(df)
    except pa.errors.SchemaError as exc:
        raise pa.errors.SchemaError(
            exc.schema, exc.data, f"{name} validation error: {exc}"
        ) from exc


__all__ = [
    "MatchesSchema",
    "Odds1x2PreSchema",
    "MarketProbsSchema",
    "validate_or_raise",
]
