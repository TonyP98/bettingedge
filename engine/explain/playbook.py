"""Utilities to aggregate and persist rule-based playbooks."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Iterable

import duckdb
import pandas as pd

from .rulefit import Rule

__all__ = ["filter_rules", "persist_rules"]


def filter_rules(
    rules: Iterable[Rule],
    *,
    min_coverage: float,
    min_precision: float,
    min_lift: float,
    max_depth: int,
    fold: int,
) -> pd.DataFrame:
    """Filter rules according to KPI thresholds and return a dataframe."""
    rows: list[dict] = []
    for r in rules:
        if (
            r.coverage >= min_coverage
            and r.precision >= min_precision
            and r.lift >= min_lift
            and r.depth <= max_depth
        ):
            rows.append(
                {
                    "rule_id": r.rule_id,
                    "rule": r.rule,
                    "depth": r.depth,
                    "support": r.support,
                    "coverage": r.coverage,
                    "precision": r.precision,
                    "lift": r.lift,
                    "fold": fold,
                    "created_at": datetime.utcnow(),
                }
            )
    return pd.DataFrame(rows)


def persist_rules(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Persist playbook rules into a DuckDB table."""
    if df.empty:
        return
    conn.execute(
        "CREATE TABLE IF NOT EXISTS explain_rules AS SELECT * FROM df LIMIT 0"
    )
    conn.execute("INSERT INTO explain_rules SELECT * FROM df")
