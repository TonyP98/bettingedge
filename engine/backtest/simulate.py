"""Simple backtest simulator supporting multiple markets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.market import markets


def run(
    signals: pd.DataFrame,
    bankroll: float = 1.0,
    market: str = "1x2",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run a sequential betting simulation for *market*."""

    if signals.empty:
        equity_df = pd.DataFrame(columns=["date", "market", "equity"])
        trades_df = pd.DataFrame(
            columns=["date", "match_id", "market", "selection", "odds", "stake", "pnl"]
        )
        metrics = {
            "roi": 0.0,
            "maxdd": 0.0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
        }
        return equity_df, trades_df, metrics

    spec = markets.MARKETS[market]
    df = signals.sort_values("date").reset_index(drop=True)

    equity = bankroll
    equity_rows = []
    trade_rows = []

    for _, row in df.iterrows():
        aux = row.get("result_aux", {})
        ft_home = aux.get("ft_home_goals")
        ft_away = aux.get("ft_away_goals")
        ah_line = aux.get("ah_line")
        payoff = spec.settle_fn(ft_home, ft_away, row["selection"], row["odds"], ah_line)
        pnl = row["stake"] * payoff
        equity += pnl

        equity_rows.append({"date": row["date"], "market": market, "equity": equity})
        trade_rows.append(
            {
                "date": row["date"],
                "match_id": row.get("match_id"),
                "market": market,
                "selection": row["selection"],
                "odds": row["odds"],
                "stake": row["stake"],
                "pnl": pnl,
            }
        )

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(trade_rows).reset_index(drop=True)

    cummax = equity_df["equity"].cummax()
    maxdd = float((equity_df["equity"] / cummax - 1).min())

    returns = trades_df["pnl"] / trades_df["stake"].replace(0, np.nan)
    sharpe = 0.0
    if returns.count() > 1 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns)))

    hit_rate = float((returns > 0).mean()) if not trades_df.empty else 0.0
    turnover = float(trades_df["stake"].sum())
    profit = float(equity_df["equity"].iloc[-1] - bankroll)
    roi = profit / turnover if turnover else 0.0

    metrics = {
        "roi": roi,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "turnover": turnover,
    }

    return equity_df, trades_df, metrics


__all__ = ["run"]

