"""Data ingestion utilities for football-data CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
import duckdb
from zoneinfo import ZoneInfo
from importlib.resources import files

from .vig import remove_vig_multiplicative, remove_vig_shin
from .bookmaker_fusion import fuse_1x2
from .contracts import MatchesSchema, MarketProbsSchema, validate_or_raise

try:
    SPEC_PATH = files("engine.data.specs") / "football_data_keys.yaml"
except Exception:  # pragma: no cover - fallback for unusual setups
    SPEC_PATH = Path("engine/data/specs/football_data_keys.yaml")
DB_PATH = Path("data/processed/football.duckdb")


def ensure_dirs() -> None:
    """Ensure raw and processed directories exist."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def load_spec() -> Dict:
    with SPEC_PATH.open() as f:
        return yaml.safe_load(f)


def parse_event_time(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    tz = ZoneInfo(spec["constraints"]["timezone"])
    date_fmt = "%d/%m/%Y %H:%M"
    event_time = pd.to_datetime(
        df["Date"].astype(str) + " " + df.get("Time", "00:00"),
        format=date_fmt,
        errors="coerce",
    ).dt.tz_localize(tz)
    df["event_time"] = event_time.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["MatchId"] = (
        event_time.dt.strftime("%Y-%m-%d")
        + "_"
        + df["HomeTeam"].astype(str)
        + "_"
        + df["AwayTeam"].astype(str)
    )
    return df


def _cast_numeric(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    min_odds = spec["constraints"]["min_odds"]
    max_odds = spec["constraints"]["max_odds"]
    for col in df.columns:
        if col == "MatchId":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < min_odds) | (df[col] > max_odds), col] = pd.NA
    return df


def parse_results(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    cols = spec["results"]["required"] + [
        c for c in spec["results"].get("optional", []) if c in df.columns
    ]
    res = df[["MatchId", "event_time"] + cols].copy()
    return res


def _extract_pre_close(df: pd.DataFrame, prefix: str) -> List[str]:
    import re

    if prefix == "pre":
        pattern = re.compile(r"^[A-Za-z0-9]+[HDA]$")
    else:
        pattern = re.compile(r"^[A-Za-z0-9]+C[HDA]$")
    return [c for c in df.columns if pattern.match(c)]


def parse_odds(df: pd.DataFrame, spec: Dict):
    pre_cols = _extract_pre_close(df, "pre")
    close_cols = _extract_pre_close(df, "close")
    odds_pre = _cast_numeric(df[["MatchId"] + pre_cols].copy(), spec)
    odds_close = None
    if close_cols:
        odds_close = _cast_numeric(df[["MatchId"] + close_cols].copy(), spec)
    return odds_pre, odds_close


def parse_ou_ah(df: pd.DataFrame, spec: Dict):
    ou_pre_cols = [c for c in spec["ou_25_pre"]["fields"] if c in df.columns]
    ou_close_cols = [c for c in spec["ou_25_close"]["fields"] if c in df.columns]
    ah_pre_cols = [
        c for c in spec["asian_handicap_pre"]["fields_any"] if c in df.columns
    ]
    ah_close_cols = [
        c for c in spec["asian_handicap_close"]["fields_any"] if c in df.columns
    ]

    ou_pre = (
        _cast_numeric(df[["MatchId"] + ou_pre_cols].copy(), spec)
        if ou_pre_cols
        else None
    )
    ou_close = (
        _cast_numeric(df[["MatchId"] + ou_close_cols].copy(), spec)
        if ou_close_cols
        else None
    )
    ah_pre = (
        _cast_numeric(df[["MatchId"] + ah_pre_cols].copy(), spec)
        if ah_pre_cols
        else None
    )
    ah_close = (
        _cast_numeric(df[["MatchId"] + ah_close_cols].copy(), spec)
        if ah_close_cols
        else None
    )
    return ou_pre, ou_close, ah_pre, ah_close


def compute_market_probs(
    matches: pd.DataFrame, odds_pre: pd.DataFrame, odds_close: pd.DataFrame | None
) -> Dict[str, pd.DataFrame]:
    tables = {}

    if {"AvgH", "AvgD", "AvgA"}.issubset(odds_pre.columns):
        fused = odds_pre.apply(
            lambda r: fuse_1x2([(r["AvgH"], r["AvgD"], r["AvgA"])])[:3],
            axis=1,
            result_type="expand",
        )
    else:
        pre_cols = [c for c in odds_pre.columns if c != "MatchId"]

        def fuse_row(r):
            books: Dict[str, Dict[str, float]] = {}
            for c in pre_cols:
                book = c[:-1]
                outcome = c[-1]
                books.setdefault(book, {})[outcome] = r[c]
            obs = [
                (vals["H"], vals["D"], vals["A"])
                for vals in books.values()
                if all(k in vals and pd.notna(vals[k]) for k in "HDA")
            ]
            if not obs:
                return pd.Series([pd.NA, pd.NA, pd.NA])
            fused = fuse_1x2(obs)
            return pd.Series(fused[:3])

        fused = odds_pre.apply(fuse_row, axis=1)

    base = pd.DataFrame(
        {"MatchId": odds_pre["MatchId"], "H": fused[0], "D": fused[1], "A": fused[2]}
    )

    probs_mul = base.apply(
        lambda r: remove_vig_multiplicative(r["H"], r["D"], r["A"]),
        axis=1,
        result_type="expand",
    )
    probs_shin = base.apply(
        lambda r: remove_vig_shin(r["H"], r["D"], r["A"]), axis=1, result_type="expand"
    )
    overround = base.apply(lambda r: sum(1.0 / r[["H", "D", "A"]]), axis=1)
    market_disagreement = (probs_mul - probs_shin).abs().sum(axis=1)

    market_probs_pre = pd.DataFrame(
        {
            "MatchId": base["MatchId"],
            "pH_mul": probs_mul[0],
            "pD_mul": probs_mul[1],
            "pA_mul": probs_mul[2],
            "pH_shin": probs_shin[0],
            "pD_shin": probs_shin[1],
            "pA_shin": probs_shin[2],
            "overround_pre": overround,
            "market_disagreement": market_disagreement,
        }
    )
    tables["market_probs_pre"] = market_probs_pre

    if odds_close is not None and {"AvgCH", "AvgCD", "AvgCA"}.issubset(
        odds_close.columns
    ):
        fused_close = odds_close.apply(
            lambda r: fuse_1x2([(r["AvgCH"], r["AvgCD"], r["AvgCA"])])[:3],
            axis=1,
            result_type="expand",
        )
        base = pd.DataFrame(
            {
                "MatchId": odds_close["MatchId"],
                "H": fused_close[0],
                "D": fused_close[1],
                "A": fused_close[2],
            }
        )
        probs_mul = base.apply(
            lambda r: remove_vig_multiplicative(r["H"], r["D"], r["A"]),
            axis=1,
            result_type="expand",
        )
        probs_shin = base.apply(
            lambda r: remove_vig_shin(r["H"], r["D"], r["A"]),
            axis=1,
            result_type="expand",
        )
        overround = base.apply(lambda r: sum(1.0 / r[["H", "D", "A"]]), axis=1)
        market_disagreement = (probs_mul - probs_shin).abs().sum(axis=1)
        market_probs_close = pd.DataFrame(
            {
                "MatchId": base["MatchId"],
                "pH_mul": probs_mul[0],
                "pD_mul": probs_mul[1],
                "pA_mul": probs_mul[2],
                "pH_shin": probs_shin[0],
                "pD_shin": probs_shin[1],
                "pA_shin": probs_shin[2],
                "overround_close": overround,
                "market_disagreement": market_disagreement,
            }
        )
        tables["market_probs_close"] = market_probs_close
    return tables


def save_tables(tables: Dict[str, pd.DataFrame]) -> None:
    conn = duckdb.connect(DB_PATH)
    try:
        for name, df in tables.items():
            conn.register(name, df)
            conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}")
    finally:
        conn.close()


def ingest(source: str, commit: bool) -> Dict[str, pd.DataFrame]:
    ensure_dirs()
    spec = load_spec()
    df = pd.read_csv(source)
    df = df.dropna(axis=1, how="all")
    df = parse_event_time(df, spec)
    results = parse_results(df, spec)
    validate_or_raise(results, MatchesSchema, "matches")
    odds_pre, odds_close = parse_odds(df, spec)
    ou_pre, ou_close, ah_pre, ah_close = parse_ou_ah(df, spec)
    tables = {
        "matches": results,
        "odds_1x2_pre": odds_pre,
    }
    if odds_close is not None:
        tables["odds_1x2_close"] = odds_close
    if ou_pre is not None:
        tables["ou_25_pre"] = ou_pre
    if ou_close is not None:
        tables["ou_25_close"] = ou_close
    if ah_pre is not None:
        tables["ah_pre"] = ah_pre
    if ah_close is not None:
        tables["ah_close"] = ah_close

    tables.update(compute_market_probs(results, odds_pre, odds_close))
    if "market_probs_pre" in tables:
        validate_or_raise(
            tables["market_probs_pre"][["pH_mul", "pD_mul", "pA_mul"]].rename(
                columns={"pH_mul": "pH", "pD_mul": "pD", "pA_mul": "pA"}
            ),
            MarketProbsSchema,
            "market_probs_pre",
        )

    if commit:
        save_tables(tables)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw data sources.")
    parser.add_argument("--source", required=True, help="CSV file to ingest")
    parser.add_argument("--commit", action="store_true", help="persist to duckdb")
    parser.add_argument(
        "--dry-run", action="store_true", help="print schema and sample"
    )
    args = parser.parse_args()

    try:
        tables = ingest(args.source, commit=args.commit)
    except Exception as e:  # pragma: no cover - CLI path
        print(f"Validation error: {e}")
        raise SystemExit(1)
    if args.dry_run:
        for name, df in tables.items():
            print(f"=== {name} ({len(df)} rows) ===")
            print(df.head())
    else:
        print(f"Ingested {len(tables['matches'])} matches from {args.source}")


if __name__ == "__main__":  # pragma: no cover
    main()
