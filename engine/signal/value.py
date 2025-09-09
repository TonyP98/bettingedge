"""Computation of value-based betting signals for multiple markets."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from engine.market import markets


def compute_value(prob: float, odds: float) -> float:
    """Expected value for binary outcome markets."""

    return prob * odds - 1.0


def _compute_ah_ev(prob_map: Mapping[str, float], odds: float) -> float:
    """Expected value for Asian handicap selections."""

    return (
        prob_map.get("win", 0.0) * (odds - 1)
        + prob_map.get("half_win", 0.0) * (odds - 1) * 0.5
        + prob_map.get("half_loss", 0.0) * (-0.5)
        + prob_map.get("loss", 0.0) * (-1.0)
    )


def _synthetic_dnb_odds(row: pd.Series) -> Mapping[str, float] | None:
    p1 = 1 / row["odds_1"]
    px = 1 / row["odds_x"]
    p2 = 1 / row["odds_2"]
    total = p1 + px + p2
    if total == 0:
        return None
    p1 /= total
    p2 /= total
    denom = p1 + p2
    if denom == 0:
        return None
    return {
        "home": 1 / (p1 / denom),
        "away": 1 / (p2 / denom),
    }


def _synthetic_dc_odds(row: pd.Series) -> Mapping[str, float] | None:
    p1 = 1 / row["odds_1"]
    px = 1 / row["odds_x"]
    p2 = 1 / row["odds_2"]
    total = p1 + px + p2
    if total == 0:
        return None
    p1 /= total
    px /= total
    p2 /= total
    return {
        "1x": 1 / (p1 + px),
        "12": 1 / (p1 + p2),
        "x2": 1 / (px + p2),
    }


def make_signals(
    df_probs: pd.DataFrame,
    ev_min: float = 0.0,
    odds_min: float | None = None,
    odds_max: float | None = None,
    max_per_day: int | None = None,
    market: str = "1x2",
    use_model_probs: str = "market",
) -> pd.DataFrame:
    """Create a long-form DataFrame of value signals for *market*."""

    spec = markets.MARKETS[market]
    df = df_probs.copy()
    df["date"] = pd.to_datetime(df["date"])

    records: list[dict] = []

    for _, row in df.iterrows():
        model_probs = {"p1": row.get("p1"), "px": row.get("px"), "p2": row.get("p2")}
        extra = {
            "lambda_home": row.get("lambda_home"),
            "lambda_away": row.get("lambda_away"),
            "ah_line": row.get("ah_line"),
        }
        prob_map = spec.prob_fn(row, model_probs, extra)

        odds_map: Mapping[str, float]
        missing = [c not in row or pd.isna(row[c]) for c in spec.odds_cols.values()]
        if any(missing):
            if market == "dnb":
                odds_map = _synthetic_dnb_odds(row) or {}
            elif market == "dc":
                odds_map = _synthetic_dc_odds(row) or {}
            else:
                odds_map = {}
        else:
            odds_map = {sel: row[col] for sel, col in spec.odds_cols.items()}

        if len(odds_map) != len(spec.targets):
            continue

        p_market = {sel: 1 / odds_map[sel] for sel in spec.targets}
        total_market = sum(p_market.values())
        if total_market == 0:
            continue
        p_market = {sel: p_market[sel] / total_market for sel in spec.targets}

        for sel in spec.targets:
            odds = odds_map[sel]
            if odds_min is not None and odds < odds_min:
                continue
            if odds_max is not None and odds > odds_max:
                continue

            if market == "ah":
                sel_probs = prob_map.get(sel, {})
                if not sel_probs:
                    continue
                p_model = sel_probs.get("win", 0.0) + 0.5 * sel_probs.get("half_win", 0.0)
                edge = _compute_ah_ev(sel_probs, odds)
            else:
                p_model = prob_map.get(sel)
                if p_model is None:
                    continue
                edge = compute_value(p_model, odds)

            if edge < ev_min:
                continue

            result_aux = {
                "ft_home_goals": row.get("ft_home_goals"),
                "ft_away_goals": row.get("ft_away_goals"),
            }
            if market == "ah":
                result_aux["ah_line"] = row.get("ah_line")

            records.append(
                {
                    "date": row["date"],
                    "home": row["home"],
                    "away": row["away"],
                    "market": market,
                    "selection": sel,
                    "odds": odds,
                    "p_model": p_model,
                    "p_market": p_market[sel],
                    "edge": edge,
                    "result_aux": result_aux,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "match_id",
                "home",
                "away",
                "market",
                "selection",
                "odds",
                "p_model",
                "p_market",
                "edge",
                "result_aux",
            ]
        )

    signals = pd.DataFrame(records)
    signals.sort_values("date", inplace=True)
    signals.insert(1, "match_id", signals.groupby(["date", "home", "away"]).ngroup())

    if max_per_day is not None:
        signals = (
            signals.groupby("date", group_keys=False)
            .apply(lambda x: x.nlargest(max_per_day, "edge"))
            .reset_index(drop=True)
        )

    return signals[
        [
            "date",
            "match_id",
            "home",
            "away",
            "market",
            "selection",
            "odds",
            "p_model",
            "p_market",
            "edge",
            "result_aux",
        ]
    ]


__all__ = ["make_signals"]

