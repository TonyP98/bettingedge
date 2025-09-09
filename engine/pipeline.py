from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from engine.data.loader import unify
from engine.io import persist
from engine.market import calibrate, odds

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def build_canonical(div: str, seasons: list[str] | None = None) -> Path:
    df = unify(div, seasons)
    out_dir = PROCESSED_DIR / div
    out_path = out_dir / "matches.parquet"
    persist.save_df(df, out_path)
    print(
        f"Saved {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()} -> {out_path}"
    )
    return out_path


def build_market(
    div: str,
    train_ratio: float = 0.8,
    train_until: str | None = None,
    calibrate_flag: bool = True,
) -> None:
    matches_path = PROCESSED_DIR / div / "matches.parquet"
    if not matches_path.exists():
        build_canonical(div)
    df = persist.load_df(matches_path)
    df["date"] = pd.to_datetime(df["date"])
    home = df["ft_home_goals"].to_numpy(dtype=float)
    away = df["ft_away_goals"].to_numpy(dtype=float)
    result = np.where(home > away, "1", np.where(home < away, "2", "x"))
    mask_missing = np.isnan(home) | np.isnan(away)
    result = pd.Series(result)
    result[mask_missing] = np.nan
    df["result"] = result
    df = df.dropna(subset=["result"])
    df_probs = odds.implied_probs(df)
    df_probs = odds.remove_vig(df_probs)
    df_probs = df_probs.dropna(subset=["p1", "px", "p2"])

    df_probs = df_probs.sort_values("date").reset_index(drop=True)
    if train_until:
        split_date = pd.to_datetime(train_until)
        train_mask = df_probs["date"] <= split_date
        df_train = df_probs[train_mask]
        df_test = df_probs[~train_mask]
    else:
        split_idx = int(len(df_probs) * train_ratio)
        df_train = df_probs.iloc[:split_idx]
        df_test = df_probs.iloc[split_idx:]
    splits = {
        "train_until": df_train["date"].max().date().isoformat() if not df_train.empty else None,
        "test_from": df_test["date"].min().date().isoformat() if not df_test.empty else None,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
    }

    out_dir = PROCESSED_DIR / div
    persist.save_json(splits, out_dir / "splits.json")

    if calibrate_flag and not df_train.empty:
        cal = calibrate.fit_calibrator(df_train)
        df_probs = calibrate.apply_calibrator(df_probs, cal)
        df_train_cal = calibrate.apply_calibrator(df_train, cal)
        report = calibrate.calibration_report(df_train_cal)
        persist.save_json(report, out_dir / "calibration_report.json")
        print(f"Train ECE: {report['ece']:.6f}")
        print(f"Train Brier: {report['brier']:.6f}")
        print(f"Train KS p-value: {report['ks_p']:.6f}")

    persist.save_df(df_probs[["date", "home", "away", "p1", "px", "p2"]], out_dir / "probs.parquet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BettingEdge pipeline")
    parser.add_argument("--rebuild-canonical", action="store_true")
    parser.add_argument("--build-market", action="store_true")
    parser.add_argument("--div", required=True)
    parser.add_argument("--seasons", nargs="*", default=[])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-until")
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    seasons = None if not args.seasons or args.seasons == ["all"] else args.seasons

    if args.rebuild_canonical:
        build_canonical(args.div, seasons)
    elif args.build_market:
        build_market(
            args.div,
            train_ratio=args.train_ratio,
            train_until=args.train_until,
            calibrate_flag=args.calibrate,
        )
    else:
        parser.error("No action specified")


if __name__ == "__main__":
    main()
