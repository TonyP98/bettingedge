from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root (contains "engine/")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from engine.data import loader
from engine.io import persist, runs
from engine.market import calibrate, odds
from engine.signal import sizing, value
from engine.backtest import simulate


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = loader.RAW_DIR


@st.cache_data
def list_divisions() -> list[str]:
    return sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())


@st.cache_data
def list_seasons(div: str) -> list[str]:
    files = sorted((RAW_DIR / div).glob("*.csv"))
    pat = re.compile(r".*_(\d{2}_\d{2})\.csv")
    seasons: list[str] = []
    for f in files:
        m = pat.match(f.name)
        if m:
            seasons.append(m.group(1))
    return seasons


@st.cache_data
def build_canonical(div: str, seasons: tuple[str, ...] | None) -> pd.DataFrame:
    return loader.unify(div, seasons)


@st.cache_data
def load_matches(div: str) -> pd.DataFrame:
    path = PROCESSED_DIR / div / "matches.parquet"
    return persist.load_df(path)


@st.cache_data
def load_probs(div: str) -> pd.DataFrame:
    path = PROCESSED_DIR / div / "probs.parquet"
    return persist.load_df(path)


def render_data_panel() -> None:
    divs = list_divisions()
    default_div = divs.index("I1") if "I1" in divs else 0
    div = st.selectbox("Div", divs, index=default_div)
    st.session_state["div"] = div

    seasons_available = list_seasons(div)
    seasons = st.multiselect("Stagioni", seasons_available, default=seasons_available)
    st.session_state["seasons"] = seasons

    if st.button("Costruisci canonico"):
        with st.spinner("Costruzione in corso..."):
            selected = (
                tuple(seasons)
                if seasons and set(seasons) != set(seasons_available)
                else None
            )
            df = build_canonical(div, selected)
            out_dir = PROCESSED_DIR / div
            persist.save_df(df, out_dir / "matches.parquet")
            load_matches.clear()

        st.write(f"Righe: {len(df)}")
        st.write(
            f"Range date: {df['date'].min().date()} - {df['date'].max().date()}"
        )
        st.session_state["canon_ok"] = True
        st.session_state["matches_df"] = df


def _build_market(
    div: str, train_ratio: float, train_until: str | None, calibrate_flag: bool
) -> tuple[dict, dict]:
    df = load_matches(div)
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
        "train_until": df_train["date"].max().date().isoformat()
        if not df_train.empty
        else None,
        "test_from": df_test["date"].min().date().isoformat()
        if not df_test.empty
        else None,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
    }

    out_dir = PROCESSED_DIR / div
    persist.save_json(splits, out_dir / "splits.json")

    if calibrate_flag and not df_train.empty:
        cal = calibrate.fit_calibrator(df_train)
        df_probs = calibrate.apply_calibrator(df_probs, cal)
        df_train_rep = calibrate.apply_calibrator(df_train, cal)
    else:
        df_train_rep = df_train

    report = calibrate.calibration_report(df_train_rep) if not df_train.empty else {}
    persist.save_json(report, out_dir / "calibration_report.json")

    persist.save_df(
        df_probs[["date", "home", "away", "p1", "px", "p2"]],
        out_dir / "probs.parquet",
    )
    load_probs.clear()

    return splits, report


def render_market_panel() -> None:
    if not st.session_state.get("canon_ok"):
        st.info("Costruisci il canonico nel pannello Dati.")
        return

    div = st.session_state.get("div", "I1")
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.8, step=0.05)
    train_until = st.text_input(
        "Data split (YYYY-MM-DD, lascia vuoto per usare il ratio)", ""
    ).strip() or None
    calibrate_flag = st.checkbox("Calibra", value=True)

    if st.button("Ricostruisci probabilità"):
        with st.spinner("Elaborazione..."):
            splits, report = _build_market(div, train_ratio, train_until, calibrate_flag)
        st.session_state["market_ok"] = True
        st.session_state["split_ok"] = True
        st.session_state["splits"] = splits
        st.session_state["calibration_report"] = report
        st.write(f"ECE: {report.get('ece', float('nan')):.6f}")
        st.write(f"Brier: {report.get('brier', float('nan')):.6f}")


def _generate_picks(
    div: str, ev_min: float, stake_mode: str, stake_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_matches = load_matches(div)
    df_probs = load_probs(div)
    df = df_matches.merge(df_probs, on=["date", "home", "away"], how="inner")
    df["date"] = pd.to_datetime(df["date"])
    home = df["ft_home_goals"].to_numpy(dtype=float)
    away = df["ft_away_goals"].to_numpy(dtype=float)
    result = np.where(home > away, "1", np.where(home < away, "2", "x"))
    df["result"] = result

    splits = st.session_state.get("splits", {})
    test_from = splits.get("test_from")
    if test_from:
        df = df[df["date"] >= pd.to_datetime(test_from)]

    signals = value.make_signals(df, ev_min=ev_min)
    signals = sizing.size_positions(signals, mode=stake_mode, fraction=stake_fraction)
    equity_df, trades_df, metrics = simulate.run(signals)
    return equity_df, trades_df, metrics


def render_backtest_panel() -> None:
    if not st.session_state.get("market_ok"):
        st.info("Ricostruisci le probabilità nel pannello Market.")
        return

    div = st.session_state.get("div", "I1")
    ev_min = st.number_input("EV_min", value=0.0, step=0.01)
    stake_mode = st.selectbox("Stake mode", ["fixed", "kelly_f"], index=0)
    stake_fraction = st.number_input("Stake fraction", value=0.01, step=0.01)

    if st.button("Genera picks & Backtest"):
        with st.spinner("Simulazione..."):
            equity_df, trades_df, metrics = _generate_picks(
                div, ev_min, stake_mode, stake_fraction
            )
        st.session_state.update(
            {
                "equity_df": equity_df,
                "trades_df": trades_df,
                "metrics": metrics,
                "picks_ok": True,
                "ev_min": ev_min,
                "stake_mode": stake_mode,
                "stake_fraction": stake_fraction,
            }
        )

    if st.session_state.get("picks_ok"):
        eq = st.session_state["equity_df"].set_index("date")
        st.line_chart(eq)
        st.dataframe(st.session_state["trades_df"])
        st.write(st.session_state["metrics"])
        csv_data = st.session_state["trades_df"].to_csv(index=False).encode("utf-8")
        json_data = st.session_state["trades_df"].to_json(orient="records")
        st.download_button("Export CSV", data=csv_data, file_name="trades.csv")
        st.download_button("Export JSON", data=json_data, file_name="trades.json")

        if st.button("Salva run"):
            splits = st.session_state.get("splits", {})
            config = {
                "div": div,
                "ev_min": st.session_state["ev_min"],
                "stake_mode": st.session_state["stake_mode"],
                "stake_fraction": st.session_state["stake_fraction"],
                "train_until": splits.get("train_until"),
                "test_from": splits.get("test_from"),
                "picks": int(len(st.session_state["trades_df"])),
            }
            run_dir = runs.create_run_dir(div)
            runs.finalize_run(
                run_dir,
                config,
                st.session_state["metrics"],
                st.session_state["equity_df"],
                st.session_state["trades_df"],
            )
            st.success(f"Run salvato in {run_dir}")


def main() -> None:
    st.title("BettingEdge")
    tabs = st.tabs(["Dati", "Market", "Backtest"])
    with tabs[0]:
        render_data_panel()
    with tabs[1]:
        render_market_panel()
    with tabs[2]:
        render_backtest_panel()


if __name__ == "__main__":
    main()

