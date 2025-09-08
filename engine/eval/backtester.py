"""Backtesting utilities for betting strategies."""
from __future__ import annotations

from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..risk.conformal import ConformalIntervalPredictor, MondrianIndexer
from ..risk.thresholds import conformal_kelly, lower_edge_rule
from .metrics import average_width, empirical_coverage, validity_gap


def apply_conformal_guard(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Apply conformal intervals and betting policy to ``df``.

    Parameters
    ----------
    df: DataFrame
        Must contain probability columns ``pH``, ``pD``, ``pA`` and odds ``oH``, ``oD``, ``oA``.
    cfg: DictConfig
        Configuration with ``conformal`` and ``policy`` sections.
    """

    p_cols = ["pH", "pD", "pA"]
    odds_cols = ["oH", "oD", "oA"]

    mondrian = MondrianIndexer(cfg.conformal.mondrian_keys)
    predictor = ConformalIntervalPredictor(
        cfg.conformal.alpha,
        mondrian,
        cfg.conformal.min_group_size,
        cfg.conformal.width_cap,
    )

    predictor.fit(df, p_cols, "y")
    intervals = predictor.predict(df, p_cols)
    df[[c + "_low" for c in p_cols]] = intervals["p_low"]
    df[[c + "_high" for c in p_cols]] = intervals["p_high"]

    best = df[p_cols].to_numpy().argmax(axis=1)
    odds = df[odds_cols].to_numpy()
    p_low = intervals["p_low"][np.arange(len(df)), best]
    p_high = intervals["p_high"][np.arange(len(df)), best]
    width = p_high - p_low
    edge_low = p_low * odds[np.arange(len(df)), best] - 1

    stakes: List[float] = []
    for i in range(len(df)):
        if lower_edge_rule(p_low[i], odds[i, best[i]], cfg.policy.edge_thr) and width[i] <= cfg.policy.max_width:
            stake = conformal_kelly(
                p_low[i],
                odds[i, best[i]],
                cfg.policy.kelly_cap,
                cfg.policy.variance_penalty,
                width[i],
            )
        else:
            stake = 0.0
        stakes.append(stake)

    df["stake"] = stakes
    df["edge_low"] = edge_low
    df["sel_cls"] = best

    return df


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover - demonstration
    """Entry point for the backtester CLI."""

    df = pd.DataFrame(
        {
            "pH": [0.4, 0.35],
            "pD": [0.3, 0.3],
            "pA": [0.3, 0.35],
            "oH": [2.1, 2.2],
            "oD": [3.1, 3.0],
            "oA": [3.2, 3.4],
            "y": [0, 1],
        }
    )
    df = apply_conformal_guard(df, cfg)
    cov = empirical_coverage(
        df[["pH_low", "pD_low", "pA_low"]].to_numpy(),
        df[["pH_high", "pD_high", "pA_high"]].to_numpy(),
        df["y"].to_numpy(),
    )
    print("coverage", cov)
    print(df)


if __name__ == "__main__":  # pragma: no cover
    main()
