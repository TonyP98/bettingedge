"""Diagnostic utilities for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "ece_multiclass",
    "reliability_table",
    "brier_decomposition",
    "pit_from_pmf",
    "pit_diagnostics",
]


def _validate_inputs(p_hat: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_hat = np.asarray(p_hat)
    y = np.asarray(y)
    if p_hat.ndim != 2:
        raise ValueError("p_hat must be 2D array")
    if len(p_hat) != len(y):
        raise ValueError("p_hat and y must have same length")
    return p_hat, y


def ece_multiclass(
    p_hat: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> float:
    """Expected calibration error for multiclass predictions."""
    p_hat, y = _validate_inputs(p_hat, y)
    confidences = p_hat.max(axis=1)
    preds = p_hat.argmax(axis=1)
    n = len(y)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(confidences, quantiles)
        # ensure unique edges
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:  # uniform
        bin_edges = np.linspace(0, 1, n_bins + 1)

    bins = np.digitize(confidences, bin_edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        acc = np.mean(preds[mask] == y[mask])
        conf = np.mean(confidences[mask])
        ece += np.abs(acc - conf) * mask.sum() / n
    return float(ece)


def reliability_table(
    p_hat: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
    labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Reliability information for each class.

    Returns a dataframe with columns ``bin``, ``cls``, ``bin_confidence``,
    ``bin_accuracy`` and ``count``.
    """
    p_hat, y = _validate_inputs(p_hat, y)
    n_samples, n_classes = p_hat.shape
    labels = list(labels) if labels is not None else list(range(n_classes))
    edges = np.linspace(0, 1, n_bins + 1)

    preds = p_hat.argmax(axis=1)
    confidences = p_hat[np.arange(n_samples), preds]
    bins = np.digitize(confidences, edges[1:-1], right=True)
    rows: List[Dict[str, object]] = []
    for k in range(n_classes):
        for b in range(n_bins):
            mask = (preds == k) & (bins == b)
            count = int(mask.sum())
            if count == 0:
                conf = 0.0
                acc = 0.0
            else:
                conf = float(confidences[mask].mean())
                acc = float((y[mask] == k).mean())
            rows.append(
                {
                    "bin": b,
                    "cls": labels[k],
                    "bin_confidence": conf,
                    "bin_accuracy": acc,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def brier_decomposition(
    p_hat: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> Dict[str, float]:
    """Decompose the multiclass Brier score.

    The decomposition follows Murphy (1973) and returns a dictionary with the
    keys ``uncertainty``, ``reliability`` and ``resolution``.
    """
    p_hat, y = _validate_inputs(p_hat, y)
    n_samples, n_classes = p_hat.shape

    o = np.zeros_like(p_hat)
    o[np.arange(n_samples), y.astype(int)] = 1.0

    base_rate = o.mean(axis=0)
    uncertainty = float(np.sum(base_rate * (1 - base_rate)))

    edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for k in range(n_classes):
        probs = p_hat[:, k]
        obs = o[:, k]
        bins = np.digitize(probs, edges[1:-1], right=True)
        for b in range(n_bins):
            mask = bins == b
            if not np.any(mask):
                continue
            p_bar = probs[mask].mean()
            o_bar = obs[mask].mean()
            weight = mask.sum() / n_samples
            reliability += weight * (p_bar - o_bar) ** 2
            resolution += weight * (o_bar - base_rate[k]) ** 2
    return {
        "uncertainty": uncertainty,
        "reliability": float(reliability),
        "resolution": float(resolution),
    }


def pit_from_pmf(
    goals_home: Iterable[int],
    goals_away: Iterable[int],
    pmf_table: np.ndarray | pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute randomized PIT values from a discrete goal PMF."""
    pmf = np.asarray(pmf_table)
    gh = np.asarray(goals_home, dtype=int)
    ga = np.asarray(goals_away, dtype=int)
    if rng is None:
        rng = np.random.default_rng()

    cdf = pmf.cumsum(axis=0).cumsum(axis=1)
    u = np.empty(len(gh), dtype=float)
    for i, (h, a) in enumerate(zip(gh, ga)):
        prev = 0.0
        if h > 0:
            prev += cdf[h - 1, -1]
        if a > 0:
            prev += cdf[h, a - 1]
        if h > 0 and a > 0:
            prev -= cdf[h - 1, a - 1]
        u[i] = prev + pmf[h, a] * rng.uniform()
    return u


def pit_diagnostics(u: Iterable[float]) -> Dict[str, float]:
    """Basic diagnostics for PIT samples."""
    u = np.asarray(u)
    ks = stats.kstest(u, "uniform")
    skew = stats.skew(u)
    kurt = stats.kurtosis(u, fisher=False)
    return {"ks_p": float(ks.pvalue), "skew": float(skew), "kurt": float(kurt)}


@dataclass
class ReliabilityBin:
    bin: int
    cls: str
    bin_confidence: float
    bin_accuracy: float
    count: int


def run_diagnostics():  # pragma: no cover - placeholder for integration
    """Run the full diagnostics pipeline (to be extended)."""
    report_dir = Path("data/processed/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "diagnostics.html"
    path.write_text("<html><body>diagnostics</body></html>")
    return str(path)
