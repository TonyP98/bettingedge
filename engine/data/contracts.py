"""Data contracts defined using Pandera schemas."""
from __future__ import annotations

try:
    import pandera as pa
    from pandera.typing import Series
except Exception:  # pragma: no cover
    pa = None  # type: ignore


if pa:
    class ResultsSchema(pa.SchemaModel):
        match_id: Series[str]
        home_goals: Series[int]
        away_goals: Series[int]

    class OddsSchema(pa.SchemaModel):
        match_id: Series[str]
        home_win: Series[float]
        draw: Series[float]
        away_win: Series[float]
        home_win_close: Series[float]
        draw_close: Series[float]
        away_win_close: Series[float]
else:  # pragma: no cover - fallback definitions
    class ResultsSchema:  # type: ignore
        """Placeholder Results schema."""

    class OddsSchema:  # type: ignore
        """Placeholder Odds schema."""

__all__ = ["ResultsSchema", "OddsSchema"]
