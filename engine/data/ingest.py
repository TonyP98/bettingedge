"""Data ingestion utilities for football-data CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yaml
import duckdb
from zoneinfo import ZoneInfo
from importlib.resources import files

log = logging.getLogger(__name__)

# === Football-Data awareness ==================================================
# Chiavi "esatte" secondo il documento "Notes for Football Data" (Football-Data.co.uk)
# Usate per riconoscere e mappare i CSV storici/attuali senza richiedere rinominhe manuali.
FD_EXACT_KEYS_RESULTS = {
    # risultati principali
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HG",
    "AG",
    "Res",
    "HTHG",
    "HTAG",
    "HTR",
}

# Alias raggruppati per campo canonico che usiamo internamente
FD_ALIASES: Dict[str, List[str]] = {
    "Date": ["Date", "DATA", "date", "Datetime", "DateTime", "TIMESTAMP", "datetime"],
    "Time": ["Time", "Ora", "HOUR", "time"],
    "Home": ["Home", "HomeTeam", "SquadraCasa", "home"],
    "Away": ["Away", "AwayTeam", "SquadraOspite", "away"],
    "HomeGoals": ["HomeGoals", "FTHG", "HG", "GolCasa", "home_goals"],
    "AwayGoals": ["AwayGoals", "FTAG", "AG", "GolOspite", "away_goals"],
    "FTR": ["FTR", "Res"],
    "HTHG": ["HTHG"],
    "HTAG": ["HTAG"],
    "HTR": ["HTR"],
    "MatchId": ["MatchId", "match_id", "matchid"],
    "event_time": ["event_time", "EventTime", "eventTime", "Datetime", "DateTime"],
}


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Trova la prima colonna presente in df tra i candidati (case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _fd_map(df: pd.DataFrame, key: str) -> Optional[str]:
    """Trova la colonna per un campo canonico usando FD_ALIASES."""
    return _find_col(df, *FD_ALIASES.get(key, []))

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
    """Parser robusto per risultati di partite di calcio.

    Riconosce alias comuni (inclusi quelli di Football-Data.co.uk) e costruisce
    automaticamente ``event_time`` e ``MatchId`` se assenti. I nomi delle
    colonne richieste sono determinati da ``spec``.
    """

    df2 = df.copy()

    # 1) Football-Data awareness
    exact_hits = FD_EXACT_KEYS_RESULTS.intersection(set(df2.columns))
    if exact_hits:
        log.info(
            "Football-Data schema detected (hits: %s). Using FD-aware mapping.",
            sorted(exact_hits),
        )
    else:
        log.warning(
            "Football-Data exact keys not detected. Proceeding with generic aliases. "
            "If your data comes from football-data.co.uk, please keep canonical headers where possible."
        )

    # 2) Individuazione colonne logiche
    c_date = _fd_map(df2, "Date")
    c_time = _fd_map(df2, "Time")
    c_dt = _find_col(df2, "Datetime", "DateTime", "datetime", "TIMESTAMP")
    c_home = _fd_map(df2, "Home")
    c_away = _fd_map(df2, "Away")
    c_hg = _fd_map(df2, "HomeGoals")
    c_ag = _fd_map(df2, "AwayGoals")
    c_ftr = _fd_map(df2, "FTR")
    c_hthg = _fd_map(df2, "HTHG")
    c_htag = _fd_map(df2, "HTAG")
    c_htr = _fd_map(df2, "HTR")
    c_mid = _fd_map(df2, "MatchId")
    c_evt = _fd_map(df2, "event_time")

    # 3) Costruzione event_time
    if c_evt is None:
        if c_dt:
            dt = pd.to_datetime(df2[c_dt], errors="coerce", dayfirst=True, utc=False)
        else:
            if c_date is None:
                raise ValueError("Missing required date/datetime columns to build event_time.")
            date_str = df2[c_date].astype(str)
            time_str = df2[c_time].astype(str) if c_time else "00:00:00"
            dt = pd.to_datetime(
                date_str + " " + time_str, errors="coerce", dayfirst=True, utc=False
            )
        df2["event_time"] = dt
        c_evt = "event_time"
    else:
        df2[c_evt] = pd.to_datetime(df2[c_evt], errors="coerce", utc=True)

    # 4) Verifiche core
    missing_core = [
        name
        for name, col in {
            "Home": c_home,
            "Away": c_away,
            "HomeGoals": c_hg,
            "AwayGoals": c_ag,
        }.items()
        if col is None
    ]
    if missing_core:
        raise KeyError(f"Missing required columns: {missing_core}")

    # 5) MatchId
    if c_mid is None:
        def mk_id(row):
            key = f"{row[c_evt]}|{row[c_home]}-{row[c_away]}"
            return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

        df2["MatchId"] = df2.apply(mk_id, axis=1)
        c_mid = "MatchId"

    # 6) Preparazione colonne richieste da spec
    results_spec = spec.get("results_cols")
    if results_spec is None:
        res_def = spec.get("results", {})
        results_spec = res_def.get("required", []) + [
            c for c in res_def.get("optional", []) if c in df2.columns
        ]
    cols_req = list(results_spec)

    out_cols: List[str] = []
    for c in cols_req:
        real = _fd_map(df2, c) or (c if c in df2.columns else None)
        if real is None:
            real = _find_col(df2, c, c.lower(), c.upper())
        if real is None:
            lc = c.lower()
            if lc == "time":
                df2["Time"] = df2[c_evt].dt.strftime("%H:%M:%S")
                real = "Time"
            elif lc == "date":
                df2["Date"] = df2[c_evt].dt.strftime("%Y-%m-%d")
                real = "Date"
            elif lc == "hometeam" and c_home:
                real = c_home
            elif lc == "awayteam" and c_away:
                real = c_away
            elif lc in ("fthg", "hg") and c_hg:
                real = c_hg
            elif lc in ("ftag", "ag") and c_ag:
                real = c_ag
            elif lc == "ftr" and c_ftr:
                real = c_ftr
            elif lc == "hthg" and c_hthg:
                real = c_hthg
            elif lc == "htag" and c_htag:
                real = c_htag
            elif lc == "htr" and c_htr:
                real = c_htr
            else:
                raise KeyError(f"Column '{c}' required by spec not found or derivable.")
        out_cols.append(real)

    # 7) Componi output canonico
    res = df2[[c_mid, c_evt] + out_cols].copy()
    res.rename(columns={c_mid: "MatchId", c_evt: "event_time"}, inplace=True)
    res["event_time"] = pd.to_datetime(res["event_time"], errors="coerce", utc=True)
    res["event_time"] = res["event_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
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


def _to_num(s: pd.Series) -> pd.Series:
    """Coerce to numeric odds, invalid values -> NaN. Discard non-sense (<=1)."""
    x = pd.to_numeric(s, errors="coerce")
    x = x.mask(~np.isfinite(x) | (x <= 1))
    return x


_BOOK_PREFIXES_1X2 = [
    # principali presenti in Football-Data (varia per stagione/lega)
    "B365",
    "BS",
    "BW",
    "GB",
    "IW",
    "LB",
    "PS",
    "SO",
    "SB",
    "SJ",
    "SY",
    "VC",
    "WH",
    "P",
]


def _derive_avg_trio(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Derive or normalise AvgH/AvgD/AvgA columns in a robust way.

    Priority order:
    1. Use existing ``AvgH/AvgD/AvgA`` columns after numeric coercion.
    2. Otherwise, compute the median across known bookmaker columns.
    3. Fallback to ``PSH/PSD/PSA`` then ``MaxH/MaxD/MaxA`` when needed.
    """

    out = odds_df.copy()

    # If Avg* columns already exist, just coerce and return
    if all(c in out.columns for c in ("AvgH", "AvgD", "AvgA")):
        out["AvgH"] = _to_num(out["AvgH"])
        out["AvgD"] = _to_num(out["AvgD"])
        out["AvgA"] = _to_num(out["AvgA"])
        return out

    # Gather bookmaker-specific columns for each outcome
    cols_H = [f"{p}H" for p in _BOOK_PREFIXES_1X2 if f"{p}H" in out.columns]
    cols_D = [f"{p}D" for p in _BOOK_PREFIXES_1X2 if f"{p}D" in out.columns]
    cols_A = [f"{p}A" for p in _BOOK_PREFIXES_1X2 if f"{p}A" in out.columns]

    def _median_across(cols: List[str]) -> pd.Series:
        if not cols:
            return pd.Series([pd.NA] * len(out), index=out.index, dtype="float64")
        stack = pd.concat([_to_num(out[c]).rename(c) for c in cols], axis=1)
        return stack.median(axis=1, skipna=True)

    avgH = _median_across(cols_H)
    avgD = _median_across(cols_D)
    avgA = _median_across(cols_A)

    # Fallback: PSH/PSD/PSA
    for src, tgt in (("PSH", "AvgH"), ("PSD", "AvgD"), ("PSA", "AvgA")):
        if src in out.columns:
            v = _to_num(out[src])
            if tgt == "AvgH":
                avgH = avgH.fillna(v)
            elif tgt == "AvgD":
                avgD = avgD.fillna(v)
            elif tgt == "AvgA":
                avgA = avgA.fillna(v)

    # Further fallback: MaxH/MaxD/MaxA
    for src, tgt in (("MaxH", "AvgH"), ("MaxD", "AvgD"), ("MaxA", "AvgA")):
        if src in out.columns:
            v = _to_num(out[src])
            if tgt == "AvgH":
                avgH = avgH.fillna(v)
            elif tgt == "AvgD":
                avgD = avgD.fillna(v)
            elif tgt == "AvgA":
                avgA = avgA.fillna(v)

    out["AvgH"], out["AvgD"], out["AvgA"] = avgH, avgD, avgA
    return out


def compute_market_probs(
    matches: pd.DataFrame, odds_pre: pd.DataFrame, odds_close: pd.DataFrame | None
) -> Dict[str, pd.DataFrame]:
    tables = {}

    # Derive/normalise AvgH/AvgD/AvgA in a robust way
    odds_pre2 = _derive_avg_trio(odds_pre)

    # Filter rows with complete trio to avoid NaNs in subsequent computations
    mask_ok = odds_pre2[["AvgH", "AvgD", "AvgA"]].notna().all(axis=1)
    if mask_ok.sum() == 0:
        raise ValueError(
            "Nessuna riga con trio di quote complete (AvgH/AvgD/AvgA). Verifica i CSV odds pre-match."
        )
    coverage = mask_ok.mean() * 100
    log.info(
        "compute_market_probs: coverage pre %.1f%% (%s/%s)",
        coverage,
        int(mask_ok.sum()),
        len(mask_ok),
    )
    pre_valid = odds_pre2.loc[mask_ok, ["MatchId", "AvgH", "AvgD", "AvgA"]]

    fused_vals = pre_valid.apply(
        lambda r: fuse_1x2([(r["AvgH"], r["AvgD"], r["AvgA"])])[:3],
        axis=1,
        result_type="expand",
    )
    fused_vals.columns = ["H", "D", "A"]

    base_valid = pd.DataFrame(
        {
            "MatchId": pre_valid["MatchId"],
            "H": fused_vals["H"],
            "D": fused_vals["D"],
            "A": fused_vals["A"],
        },
        index=pre_valid.index,
    )

    probs_mul = base_valid.apply(
        lambda r: remove_vig_multiplicative(r["H"], r["D"], r["A"]),
        axis=1,
        result_type="expand",
    )
    probs_shin = base_valid.apply(
        lambda r: remove_vig_shin(r["H"], r["D"], r["A"]),
        axis=1,
        result_type="expand",
    )
    overround = base_valid.apply(lambda r: sum(1.0 / r[["H", "D", "A"]]), axis=1)
    market_disagreement = (probs_mul - probs_shin).abs().sum(axis=1)

    market_probs_pre_valid = pd.DataFrame(
        {
            "pH_mul": probs_mul[0],
            "pD_mul": probs_mul[1],
            "pA_mul": probs_mul[2],
            "pH_shin": probs_shin[0],
            "pD_shin": probs_shin[1],
            "pA_shin": probs_shin[2],
            "overround_pre": overround,
            "market_disagreement": market_disagreement,
        },
        index=base_valid.index,
    )

    # Reattach to original index, leaving NaNs for rows without complete trio
    market_probs_pre = pd.DataFrame({"MatchId": odds_pre2["MatchId"]})
    market_probs_pre = market_probs_pre.join(market_probs_pre_valid)
    float_cols = [c for c in market_probs_pre.columns if c != "MatchId"]
    market_probs_pre[float_cols] = market_probs_pre[float_cols].astype(float)
    tables["market_probs_pre"] = market_probs_pre

    if odds_close is not None and {"AvgCH", "AvgCD", "AvgCA"}.issubset(
        odds_close.columns
    ):
        mask_close = odds_close[["AvgCH", "AvgCD", "AvgCA"]].notna().all(axis=1)
        coverage_close = mask_close.mean() * 100
        log.info(
            "compute_market_probs: coverage close %.1f%% (%s/%s)",
            coverage_close,
            int(mask_close.sum()),
            len(mask_close),
        )
        close_valid = odds_close.loc[mask_close, ["MatchId", "AvgCH", "AvgCD", "AvgCA"]]
        fused_close = close_valid.apply(
            lambda r: fuse_1x2([(r["AvgCH"], r["AvgCD"], r["AvgCA"])])[:3],
            axis=1,
            result_type="expand",
        )
        fused_close.columns = ["H", "D", "A"]
        base = pd.DataFrame(
            {
                "MatchId": close_valid["MatchId"],
                "H": fused_close["H"],
                "D": fused_close["D"],
                "A": fused_close["A"],
            },
            index=close_valid.index,
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
        market_probs_close_valid = pd.DataFrame(
            {
                "pH_mul": probs_mul[0],
                "pD_mul": probs_mul[1],
                "pA_mul": probs_mul[2],
                "pH_shin": probs_shin[0],
                "pD_shin": probs_shin[1],
                "pA_shin": probs_shin[2],
                "overround_close": overround,
                "market_disagreement": market_disagreement,
            },
            index=base.index,
        )
        market_probs_close = pd.DataFrame({"MatchId": odds_close["MatchId"]})
        market_probs_close = market_probs_close.join(market_probs_close_valid)
        float_cols = [c for c in market_probs_close.columns if c != "MatchId"]
        market_probs_close[float_cols] = market_probs_close[float_cols].astype(float)
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
    df = df.dropna(axis=1, how="all").dropna(how="all").convert_dtypes()
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

    if "market_probs_close" in tables:
        validate_or_raise(
            tables["market_probs_close"][["pH_mul", "pD_mul", "pA_mul"]].rename(
                columns={"pH_mul": "pH", "pD_mul": "pD", "pA_mul": "pA"}
            ),
            MarketProbsSchema,
            "market_probs_close",
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
