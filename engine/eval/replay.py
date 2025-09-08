"""Walk-forward replay simulator for bandit policies."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..online.context import build_context_row, encode_context
from ..online.bandits import BanditBase


def crra_reward(pnl: float, bankroll: float, gamma: float) -> float:
    """Constant Relative Risk Aversion (CRRA) utility clipped to [-1, 1]."""

    u = ((1 + pnl / bankroll) ** (1 - gamma) - 1) / (1 - gamma)
    return float(np.clip(u, -1.0, 1.0))


def replay_bandit(df: pd.DataFrame, bandit: BanditBase, cfg) -> pd.DataFrame:
    """Run a walk-forward replay using ``bandit`` over ``df``.

    Parameters
    ----------
    df : DataFrame
        Must contain at least ``match_id`` and a reward ``pnl`` column.
    bandit : BanditBase
        The bandit agent to evaluate.
    cfg : Any
        Configuration with ``gamma_crra`` for the reward function.
    """

    logs = []
    bankroll = 1.0
    for _, row in df.iterrows():
        ctx = build_context_row(row, {})
        X, _ = encode_context(pd.DataFrame([ctx]))
        x = X[0]
        sel = bandit.select(x)
        edge_thr, kcap = sel["action"]
        stake = row.get("stake", 0.0)
        pnl = row.get("pnl", 0.0)
        reward = crra_reward(pnl, bankroll, cfg.gamma_crra)
        bandit.update(x, sel["action"], reward)
        bankroll += pnl
        logs.append(
            {
                "t": int(row.get("t", 0)),
                "match_id": row.get("match_id", ""),
                "action_edge_thr": edge_thr,
                "action_kcap": kcap,
                "context": ctx,
                "reward": reward,
                "stake": stake,
                "pnl": pnl,
                "bankroll": bankroll,
                "algo": bandit.__class__.__name__,
                "created_at": pd.Timestamp.utcnow(),
            }
        )
    return pd.DataFrame(logs)
