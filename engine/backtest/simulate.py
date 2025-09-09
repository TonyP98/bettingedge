"""Simple backtest simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run(
    signals: pd.DataFrame,
    bankroll: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run a sequential betting simulation.

    Parameters
    ----------
    signals:
        DataFrame with at least the columns ``date``, ``home``, ``away``,
        ``selection``, ``odds``, ``stake`` and ``result``.
    bankroll:
        Starting bankroll. Defaults to 1 unit.
    """

    if signals.empty:
        equity_df = pd.DataFrame(columns=["date", "equity"])
        trades_df = pd.DataFrame(
            columns=[
                "date",
                "match",
                "selection",
                "odds",
                "stake",
                "result",
                "pnl",
                "equity",
            ]
        )
        metrics = {
            "cagr": 0.0,
            "maxdd": 0.0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
        }
        return equity_df, trades_df, metrics

    df = signals.sort_values("date").reset_index(drop=True)

    equity = bankroll
    equity_rows = []
    trade_rows = []

    for _, row in df.iterrows():
        hit = row["result"] == row["selection"]
        pnl = row["stake"] * (row["odds"] - 1) if hit else -row["stake"]
        equity += pnl

        equity_rows.append({"date": row["date"], "equity": equity})
        trade_rows.append(
            {
                "date": row["date"],
                "match": f"{row['home']} vs {row['away']}",
                "selection": row["selection"],
                "odds": row["odds"],
                "stake": row["stake"],
                "result": int(hit),
                "pnl": pnl,
                "equity": equity,
            }
        )

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(trade_rows).reset_index(drop=True)

    days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days or 1
    cagr = (equity_df["equity"].iloc[-1] / bankroll) ** (365 / days) - 1
    cummax = equity_df["equity"].cummax()
    maxdd = float((equity_df["equity"] / cummax - 1).min())

    returns = trades_df["pnl"] / trades_df["stake"].replace(0, np.nan)
    sharpe = 0.0
    if returns.count() > 1 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns)))

    hit_rate = float(trades_df["result"].mean()) if not trades_df.empty else 0.0
    turnover = float(trades_df["stake"].sum())

    metrics = {
        "cagr": float(cagr),
        "maxdd": maxdd,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "turnover": turnover,
    }

    return equity_df, trades_df, metrics


__all__ = ["run"]

