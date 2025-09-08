"""Regime clustering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None  # type: ignore


@dataclass
class RegimeResult:
    regime_map: pd.DataFrame
    regime_kpi: pd.DataFrame


FEATURE_COLS = [
    "overround_pre",
    "avg_max_spread",
    "market_disagreement",
    "form_volatility_gap",
]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    if "month" in feats.columns:
        feats["month"] = pd.to_numeric(feats["month"], errors="coerce")
    if "Div" in feats.columns:
        feats["Div"] = feats["Div"].astype("category").cat.codes
    cols = [c for c in FEATURE_COLS if c in feats.columns]
    cols += [c for c in ["month", "Div"] if c in feats.columns]
    return feats[cols]


def _cluster(
    df: pd.DataFrame,
    algo: str = "kmeans",
    min_cluster_size: int = 50,
    k_grid: Iterable[int] | None = None,
) -> np.ndarray:
    X = df.to_numpy()
    if algo == "hdbscan" and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(X)
    elif algo == "dbscan":
        clusterer = DBSCAN(min_samples=min_cluster_size)
        labels = clusterer.fit_predict(X)
    else:
        if k_grid is None:
            k = 3
        else:
            best_k = None
            best_inertia = np.inf
            for k in k_grid:
                if k <= 1:
                    continue
                km = KMeans(n_clusters=k, n_init=10, random_state=0)
                km.fit(X)
                if km.inertia_ < best_inertia:
                    best_inertia = km.inertia_
                    best_k = k
            k = best_k or 2
        clusterer = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = clusterer.fit_predict(X)
    return labels


def regime_report(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    bets: pd.DataFrame | None = None,
    algo: str = "kmeans",
    min_cluster_size: int = 50,
    k_grid: Iterable[int] | None = None,
) -> RegimeResult:
    """Cluster matches into regimes and compute simple KPIs."""
    feats = _prepare_features(df)
    labels = _cluster(feats, algo, min_cluster_size, k_grid)
    regime_map = df.copy()
    regime_map["regime_id"] = labels

    n_samples = len(df)
    probs = preds[["pH", "pD", "pA"]].to_numpy()
    y_idx = df["y"].to_numpy()
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), y_idx] = 1
    brier = np.mean((probs - one_hot) ** 2, axis=1)
    logloss = -np.log(np.clip(probs[np.arange(n_samples), y_idx], 1e-15, 1))

    rows: List[Dict[str, object]] = []
    for rid in np.unique(labels):
        mask = labels == rid
        n = int(mask.sum())
        if n == 0:
            continue
        roi = 0.0
        if bets is not None and {"pnl", "stake"}.issubset(bets.columns):
            pnl = bets.loc[mask, "pnl"].sum()
            stake = bets.loc[mask, "stake"].sum()
            roi = float(pnl / stake) if stake > 0 else 0.0
        rows.append(
            {
                "regime_id": int(rid),
                "n": n,
                "kpi_json": {
                    "logloss": float(logloss[mask].mean()),
                    "brier": float(brier[mask].mean()),
                    "roi": roi,
                },
            }
        )
    regime_kpi = pd.DataFrame(rows)
    return RegimeResult(regime_map=regime_map[["regime_id"]], regime_kpi=regime_kpi)


__all__ = ["regime_report", "RegimeResult"]
