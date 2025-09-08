"""Evaluation metrics for conformal prediction."""
from __future__ import annotations

import numpy as np


def empirical_coverage(p_low: np.ndarray, p_high: np.ndarray, y_true: np.ndarray) -> float:
    """Empirical coverage of the lower-bound argmax strategy."""

    pred = p_low.argmax(axis=1)
    return float((pred == y_true).mean())


def average_width(p_low: np.ndarray, p_high: np.ndarray) -> float:
    """Average interval width across classes."""

    return float(np.mean((p_high - p_low).mean(axis=1)))


def validity_gap(coverage: float, alpha: float) -> float:
    """Absolute gap between empirical and target coverage."""

    return abs(coverage - (1 - alpha))
