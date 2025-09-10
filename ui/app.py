from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (contains "engine/")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import pipeline
from engine.data import loader
from engine.io import persist, runs
from engine.signal import sizing, value
from engine.backtest import simulate
from engine.market import odds, calibrate
from engine.model import poisson


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


def _parse_uploaded_matches(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    rename_map = {
        "Div": "div",
        "Date": "date",
        "Time": "time",
        "HomeTeam": "home",
        "AwayTeam": "away",
        "B365H": "odds_1",
        "B365D": "odds_x",
        "B365A": "odds_2",
        "FTHG": "ft_home_goals",
        "FTAG": "ft_away_goals",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if not {"odds_1", "odds_x", "odds_2"}.issubset(df.columns):
        lower = {c.lower(): c for c in df.columns}
        for suf, new in [("h", "odds_1"), ("d", "odds_x"), ("a", "odds_2")]:
            if new in df.columns:
                continue
            match = next((orig for lc, orig in lower.items() if lc.endswith(suf)), None)
            if match:
                df = df.rename(columns={match: new})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in ["ft_home_goals", "ft_away_goals"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


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
        st.write(f"Range date: {df['date'].min().date()} - {df['date'].max().date()}")
        st.session_state["canon_ok"] = True
        st.session_state["matches_df"] = df

    uploaded = st.file_uploader("Carica partite per il test (opzionale)")
    if uploaded is not None:
        df_up = _parse_uploaded_matches(uploaded)
        st.session_state["uploaded_df"] = df_up
        st.write(f"Partite caricate: {len(df_up)}")


def _build_market(
    div: str,
    train_ratio: float,
    train_until: str | None,
    calibrate_flag: bool,
    model_source: str,
) -> tuple[dict, dict, dict, calibrate.Calibrator | None]:
    splits, report, rates, cal = pipeline.build_market(
        div,
        train_ratio=train_ratio,
        train_until=train_until,
        calibrate_flag=calibrate_flag,
        model_source=model_source,
    )
    load_probs.clear()
    return splits, report, rates, cal


def render_market_panel() -> None:
    if not st.session_state.get("canon_ok"):
        st.info("Costruisci il canonico nel pannello Dati.")
        return

    div = st.session_state.get("div", "I1")
    uploaded_df = st.session_state.get("uploaded_df")
    disabled = uploaded_df is not None
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.8, step=0.05, disabled=disabled)
    train_until = (
        st.text_input(
            "Data split (YYYY-MM-DD, lascia vuoto per usare il ratio)", "", disabled=disabled
        ).strip()
        or None
    )
    if disabled:
        train_ratio = 1.0
        train_until = None
    calibrate_flag = st.checkbox("Calibra", value=True)
    model_source = st.selectbox(
        "Modello", ["market", "poisson", "blend(0.5)"], index=0
    ).split("(")[0]

    if st.button("Ricostruisci probabilità"):
        with st.spinner("Elaborazione..."):
            splits, report, rates, cal = _build_market(
                div, train_ratio, train_until, calibrate_flag, model_source
            )
        st.session_state["market_ok"] = True
        st.session_state["split_ok"] = True
        st.session_state["splits"] = splits
        st.session_state["calibration_report"] = report
        st.session_state["model_source"] = model_source
        st.session_state["rates"] = rates
        st.session_state["calibrator"] = cal
        st.write(f"ECE: {report.get('ece', float('nan')):.6f}")
        st.write(f"Brier: {report.get('brier', float('nan')):.6f}")


def _generate_picks(
    div: str,
    markets: list[str],
    ev_min: float,
    stake_mode: str,
    stake_fraction: float,
    bankroll: float,
    df_external: pd.DataFrame | None = None,
    rates: dict | None = None,
    calibrator_obj: calibrate.Calibrator | None = None,
    model_source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], pd.DataFrame]:
    model_source = model_source or st.session_state.get("model_source", "market")
    if df_external is None:
        df_matches = load_matches(div)
        df_probs = load_probs(div)
        df = df_matches.merge(df_probs, on=["date", "home", "away"], how="inner")
        df["date"] = pd.to_datetime(df["date"])
        splits = st.session_state.get("splits", {})
        test_from = splits.get("test_from")
        if test_from:
            df = df[df["date"] >= pd.to_datetime(test_from)]
    else:
        df = df_external.copy()
        df["date"] = pd.to_datetime(df["date"])
        df_probs = odds.implied_probs(df)
        df_probs = odds.remove_vig(df_probs)
        df_poi = poisson.predict_match_probs(df_probs[["home", "away"]], rates or {})
        if model_source == "poisson":
            df_probs[["p1", "px", "p2"]] = df_poi[["p1", "px", "p2"]]
        elif model_source == "blend":
            blended = 0.5 * df_probs[["p1", "px", "p2"]] + 0.5 * df_poi[["p1", "px", "p2"]]
            df_probs[["p1", "px", "p2"]] = blended.div(blended.sum(axis=1), axis=0)
        df_probs[["lambda_home", "lambda_away"]] = df_poi[["lambda_home", "lambda_away"]]
        if calibrator_obj is not None:
            df_probs = calibrate.apply_calibrator(df_probs, calibrator_obj)
        df = df.merge(
            df_probs[["p1", "px", "p2", "lambda_home", "lambda_away"]],
            left_index=True,
            right_index=True,
        )

    equity_dfs: list[pd.DataFrame] = []
    trades_dfs: list[pd.DataFrame] = []
    metrics_by_market: dict[str, dict] = {}
    signals = pd.DataFrame()

    results_available = set(["ft_home_goals", "ft_away_goals"]).issubset(df.columns) and df[["ft_home_goals", "ft_away_goals"]].notna().all().all()

    for mkt in markets:
        sigs = value.make_signals(
            df,
            ev_min=ev_min,
            market=mkt,
            use_model_probs=model_source,
        )
        sigs = sizing.size_positions(
            sigs, mode=stake_mode, fraction=stake_fraction, bankroll=bankroll
        )
        signals = pd.concat([signals, sigs], ignore_index=True)
        if results_available:
            eq_df, tr_df, met = simulate.run(sigs, bankroll=bankroll, market=mkt)
            tr_df = tr_df.sort_values("date").reset_index(drop=True)
            eq_from_trades = tr_df[["date"]].copy()
            eq_from_trades["market"] = mkt
            eq_from_trades["equity"] = bankroll + tr_df["pnl"].cumsum()
            equity_dfs.append(eq_from_trades)
            trades_dfs.append(tr_df)
            metrics_by_market[mkt] = met

    equity_df = pd.concat(equity_dfs, ignore_index=True) if equity_dfs else pd.DataFrame()
    trades_df = pd.concat(trades_dfs, ignore_index=True) if trades_dfs else pd.DataFrame()
    return equity_df, trades_df, metrics_by_market, signals

def render_backtest_panel() -> None:
    if not st.session_state.get("market_ok"):
        st.info("Ricostruisci le probabilità nel pannello Market.")
        return

    div = st.session_state.get("div", "I1")
    ev_min = st.number_input("EV_min", value=0.0, step=0.01)
    stake_mode = st.selectbox("Stake mode", ["fixed", "kelly_f"], index=0)
    stake_fraction = st.number_input("Stake fraction", value=0.01, step=0.01)
    bankroll = st.number_input("Equity iniziale", value=1.0, step=0.1)
    markets = st.multiselect(
        "Mercati", ["1x2", "ou25", "dnb", "dc", "ah"], default=["1x2"]
    )

    if st.button("Genera picks & Backtest"):
        df_ext = st.session_state.get("uploaded_df")
        with st.spinner("Simulazione..."):
            equity_df, trades_df, metrics, signals = _generate_picks(
                div,
                markets,
                ev_min,
                stake_mode,
                stake_fraction,
                bankroll,
                df_external=df_ext,
                rates=st.session_state.get("rates"),
                calibrator_obj=st.session_state.get("calibrator"),
                model_source=st.session_state.get("model_source"),
            )
        st.session_state.update(
            {
                "equity_df": equity_df,
                "trades_df": trades_df,
                "metrics": metrics,
                "signals": signals,
                "picks_ok": True,
                "ev_min": ev_min,
                "stake_mode": stake_mode,
                "stake_fraction": stake_fraction,
                "bankroll": bankroll,
                "markets": markets,
            }
        )

    if st.session_state.get("picks_ok"):
        if st.session_state["trades_df"].empty:
            st.dataframe(st.session_state["signals"])
            res_file = st.file_uploader("Carica risultati finali")
            if res_file is not None:
                df_res = _parse_uploaded_matches(res_file)
                with st.spinner("Backtest..."):
                    equity_df, trades_df, metrics, signals = _generate_picks(
                        div,
                        st.session_state.get("markets", markets),
                        ev_min,
                        stake_mode,
                        stake_fraction,
                        bankroll,
                        df_external=df_res,
                        rates=st.session_state.get("rates"),
                        calibrator_obj=st.session_state.get("calibrator"),
                        model_source=st.session_state.get("model_source"),
                    )
                st.session_state.update(
                    {
                        "equity_df": equity_df,
                        "trades_df": trades_df,
                        "metrics": metrics,
                        "signals": signals,
                    }
                )
                st.rerun
        else:
            eq_df = st.session_state["equity_df"].sort_values("date")
            eq_pivot = eq_df.pivot_table(
                index="date",
                columns="market",
                values="equity",
                aggfunc="last",
            )
            st.line_chart(eq_pivot)
            st.dataframe(st.session_state["trades_df"])
            st.write(pd.DataFrame(st.session_state["metrics"]).T)
            csv_data = st.session_state["trades_df"].to_csv(index=False).encode("utf-8")
            json_data = st.session_state["trades_df"].to_json(orient="records")
            st.download_button("Export CSV", data=csv_data, file_name="trades.csv")
            st.download_button("Export JSON", data=json_data, file_name="trades.json")

            if st.button("Salva run"):
                splits = st.session_state.get("splits", {})
                run_dir = runs.create_run_dir(div)
                config = {
                    "div": div,
                    "ev_min": st.session_state["ev_min"],
                    "stake_mode": st.session_state["stake_mode"],
                    "stake_fraction": st.session_state["stake_fraction"],
                    "bankroll": st.session_state["bankroll"],
                    "train_until": splits.get("train_until"),
                    "test_from": splits.get("test_from"),
                }
                persist.save_json(config, run_dir / "config.json")
                metrics_df = pd.DataFrame(st.session_state["metrics"]).T
                persist.save_df(metrics_df, run_dir / "metrics_by_market.csv")
                for mkt in st.session_state["markets"]:
                    eq_m = st.session_state["equity_df"][
                        st.session_state["equity_df"]["market"] == mkt
                    ]
                    tr_m = st.session_state["trades_df"][
                        st.session_state["trades_df"]["market"] == mkt
                    ]
                    persist.save_df(eq_m, run_dir / f"equity_{mkt}.csv")
                    persist.save_df(tr_m, run_dir / f"trades_{mkt}.csv")
                    persist.save_json(
                        st.session_state["metrics"][mkt], run_dir / f"metrics_{mkt}.json"
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
