"""Edge-level evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score

__all__ = [
    "edge_precision_recall",
    "edge_gain_curve",
    "stability_by_regime",
]


def edge_precision_recall(y_true: np.ndarray, edge_pred: np.ndarray, thr: float = 0.0):
    """Compute precision and recall for positive edge predictions."""
    y_hat = edge_pred >= thr
    precision = precision_score(y_true, y_hat, zero_division=0)
    recall = recall_score(y_true, y_hat, zero_division=0)
    return precision, recall


def edge_gain_curve(y_true: np.ndarray, edge_pred: np.ndarray) -> np.ndarray:
    """Return cumulative gain curve ordering by predicted edge."""
    order = np.argsort(-edge_pred)
    cum_gain = np.cumsum(y_true[order])
    if cum_gain.size:
        cum_gain = cum_gain / cum_gain[-1]
    return cum_gain


def stability_by_regime(precisions: np.ndarray) -> float:
    """Compute normalised variance of precisions across regimes."""
    precisions = np.asarray(precisions)
    if precisions.size == 0:
        return float("nan")
    mean = precisions.mean()
    var = precisions.var()
    return var / (mean * (1 - mean) + 1e-12)
