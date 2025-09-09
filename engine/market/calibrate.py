from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    method: str
    models: dict[str, object]


def fit_calibrator(df_train: pd.DataFrame, method: str = "isotonic") -> Calibrator:
    """Fit a probability calibrator on the training set."""
    outcome_idx = df_train["result"].map({"1": 0, "x": 1, "2": 2}).values
    probs = df_train[["p1", "px", "p2"]].to_numpy()

    calibrators: dict[str, object] = {}
    for i, key in enumerate(["p1", "px", "p2"]):
        X = probs[:, i]
        y = (outcome_idx == i).astype(int)
        if method == "platt":
            model = LogisticRegression(max_iter=1000)
            model.fit(X.reshape(-1, 1), y)
        else:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(X, y)
        calibrators[key] = model
    return Calibrator(method=method, models=calibrators)


def apply_calibrator(df: pd.DataFrame, cal: Calibrator) -> pd.DataFrame:
    """Apply a trained calibrator to a DataFrame of probabilities."""
    df = df.copy()
    for key in ["p1", "px", "p2"]:
        model = cal.models[key]
        X = df[key].to_numpy()
        if cal.method == "platt":
            df[key] = model.predict_proba(X.reshape(-1, 1))[:, 1]
        else:
            df[key] = model.predict(X)
    total = df[["p1", "px", "p2"]].sum(axis=1)
    df[["p1", "px", "p2"]] = df[["p1", "px", "p2"]].div(total, axis=0)
    return df


def calibration_report(df: pd.DataFrame) -> dict[str, float]:
    """Compute calibration metrics on a probability DataFrame."""
    probs = df[["p1", "px", "p2"]].to_numpy()
    outcome_idx = df["result"].map({"1": 0, "x": 1, "2": 2}).to_numpy()
    n = len(df)

    labels = np.zeros_like(probs)
    labels[np.arange(n), outcome_idx] = 1

    brier = float(np.mean(np.sum((probs - labels) ** 2, axis=1)))

    probs_flat = probs.ravel()
    labels_flat = labels.ravel()
    ece = _ece(probs_flat, labels_flat)

    actual_probs = probs[np.arange(n), outcome_idx]
    ks_p = float(kstest(actual_probs, "uniform").pvalue)

    return {"ece": ece, "brier": brier, "ks_p": ks_p}


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        mask = idx == i
        if not np.any(mask):
            continue
        bin_prob = probs[mask]
        bin_labels = labels[mask]
        acc = bin_labels.mean()
        conf = bin_prob.mean()
        ece += np.abs(acc - conf) * (mask.sum() / total)
    return float(ece)
