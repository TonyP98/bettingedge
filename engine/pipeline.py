from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from engine.data.loader import unify
from engine.io import persist, runs
from engine.market import calibrate, odds
from engine.model import poisson
from engine.signal import value, sizing
from engine.backtest import simulate

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
    model_source: str = "market",
) -> tuple[dict, dict]:
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
        train_idx = train_mask.to_numpy()
    else:
        split_idx = int(len(df_probs) * train_ratio)
        train_idx = np.zeros(len(df_probs), dtype=bool)
        train_idx[:split_idx] = True

    if model_source in {"poisson", "blend"}:
        rates = poisson.fit_team_rates(
            df_probs.loc[train_idx, ["home", "away", "ft_home_goals", "ft_away_goals"]]
        )
        df_poi = poisson.predict_match_probs(df_probs[["home", "away"]], rates)
        if model_source == "poisson":
            df_probs[["p1", "px", "p2"]] = df_poi[["p1", "px", "p2"]]
        else:
            blended = (
                0.5 * df_probs[["p1", "px", "p2"]] + 0.5 * df_poi[["p1", "px", "p2"]]
            )
            df_probs[["p1", "px", "p2"]] = blended.div(blended.sum(axis=1), axis=0)

    df_train = df_probs.loc[train_idx]
    df_test = df_probs.loc[~train_idx]
    splits = {
        "train_until": (
            df_train["date"].max().date().isoformat() if not df_train.empty else None
        ),
        "test_from": (
            df_test["date"].min().date().isoformat() if not df_test.empty else None
        ),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
    }

    out_dir = PROCESSED_DIR / div
    persist.save_json(splits, out_dir / "splits.json")

    report: dict[str, float] = {}
    if calibrate_flag and not df_train.empty:
        cal = calibrate.fit_calibrator(df_train)
        df_probs = calibrate.apply_calibrator(df_probs, cal)
        df_train_cal = calibrate.apply_calibrator(df_train, cal)
        report = calibrate.calibration_report(df_train_cal)
        persist.save_json(report, out_dir / "calibration_report.json")
        print(f"Train ECE: {report['ece']:.6f}")
        print(f"Train Brier: {report['brier']:.6f}")
        print(f"Train KS p-value: {report['ks_p']:.6f}")
    else:
        persist.save_json(report, out_dir / "calibration_report.json")

    persist.save_df(
        df_probs[["date", "home", "away", "p1", "px", "p2"]],
        out_dir / "probs.parquet",
    )

    return splits, report


def generate_picks(
    div: str,
    ev_min: float = 0.0,
    stake_mode: str = "fixed",
    stake_fraction: float = 0.01,
    save_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate value signals, size bets and simulate a backtest."""

    out_dir = PROCESSED_DIR / div
    matches_path = out_dir / "matches.parquet"
    probs_path = out_dir / "probs.parquet"
    if not matches_path.exists() or not probs_path.exists():
        raise FileNotFoundError(
            "Required data files not found. Run with --build-market first."
        )

    df_matches = persist.load_df(matches_path)
    df_probs = persist.load_df(probs_path)
    df = df_matches.merge(df_probs, on=["date", "home", "away"], how="inner")
    df["date"] = pd.to_datetime(df["date"])

    home = df["ft_home_goals"].to_numpy(dtype=float)
    away = df["ft_away_goals"].to_numpy(dtype=float)
    result = np.where(home > away, "1", np.where(home < away, "2", "x"))
    df["result"] = result

    splits_path = out_dir / "splits.json"
    splits = persist.load_json(splits_path) if splits_path.exists() else {}
    test_from = splits.get("test_from")
    if test_from:
        df = df[df["date"] >= pd.to_datetime(test_from)]

    signals = value.make_signals(df, ev_min=ev_min)
    signals = sizing.size_positions(signals, mode=stake_mode, fraction=stake_fraction)
    equity_df, trades_df, metrics = simulate.run(signals)

    if save_run:
        config = {
            "div": div,
            "ev_min": ev_min,
            "stake_mode": stake_mode,
            "stake_fraction": stake_fraction,
            "train_until": splits.get("train_until"),
            "test_from": splits.get("test_from"),
            "picks": int(len(trades_df)),
        }
        run_dir = runs.create_run_dir(div)
        runs.finalize_run(run_dir, config, metrics, equity_df, trades_df)

    return equity_df, trades_df, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BettingEdge pipeline")
    parser.add_argument("--rebuild-canonical", action="store_true")
    parser.add_argument("--build-market", action="store_true")
    parser.add_argument("--div", required=True)
    parser.add_argument("--seasons", nargs="*", default=[])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-until")
    parser.add_argument(
        "--calibrate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--model-source", choices=["market", "poisson", "blend"], default="market"
    )
    parser.add_argument("--picks", action="store_true")
    parser.add_argument("--ev-min", type=float, default=0.0)
    parser.add_argument("--stake-mode", choices=["fixed", "kelly_f"], default="fixed")
    parser.add_argument("--stake-fraction", type=float, default=0.01)
    parser.add_argument("--save-run", action="store_true")
    args = parser.parse_args(argv)

    seasons = None if not args.seasons or args.seasons == ["all"] else args.seasons

    action_performed = False
    if args.rebuild_canonical:
        build_canonical(args.div, seasons)
        action_performed = True
    if args.build_market:
        build_market(
            args.div,
            train_ratio=args.train_ratio,
            train_until=args.train_until,
            calibrate_flag=args.calibrate,
            model_source=args.model_source,
        )
        action_performed = True
    if args.picks:
        generate_picks(
            args.div,
            ev_min=args.ev_min,
            stake_mode=args.stake_mode,
            stake_fraction=args.stake_fraction,
            save_run=args.save_run,
        )
        action_performed = True
    if not action_performed:
        parser.error("No action specified")


if __name__ == "__main__":
    main()
