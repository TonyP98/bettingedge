"""Conformal prediction utilities for probability intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator

import numpy as np
import pandas as pd


@dataclass
class VennAbersCalibrator:
    """Simple split-conformal calibrator for multiclass probabilities."""

    alpha: float = 0.1
    q_: np.ndarray | None = None

    def fit(self, p_hat: np.ndarray, y: np.ndarray) -> "VennAbersCalibrator":
        """Fit nonconformity scores and quantiles.

        Parameters
        ----------
        p_hat: array-like of shape (n_samples, n_classes)
            Raw probability estimates from a classifier.
        y: array-like of shape (n_samples,)
            True class labels encoded as integers ``[0, n_classes)``.
        """

        n_classes = p_hat.shape[1]
        self.q_ = np.zeros(n_classes)
        for c in range(n_classes):
            y_bin = (y == c).astype(float)
            scores = np.abs(y_bin - p_hat[:, c])
            # (1 - alpha) quantile of nonconformity scores
            self.q_[c] = np.quantile(scores, 1 - self.alpha)
        return self

    def predict_interval(self, p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict probability intervals for each class.

        Parameters
        ----------
        p_hat: array-like of shape (n_samples, n_classes)
            Raw probability estimates for which to compute intervals.
        """

        if self.q_ is None:
            raise ValueError("Calibrator must be fitted before prediction")

        p_low = np.clip(p_hat - self.q_, 0.0, 1.0)
        p_high = np.clip(p_hat + self.q_, 0.0, 1.0)
        return p_low, p_high


class MondrianIndexer:
    """Utility to build Mondrian partitioning keys."""

    def __init__(self, keys: Iterable[str] | None = None):
        self.keys = list(keys or [])

    def make_keys(self, df_matches: pd.DataFrame) -> np.ndarray:
        """Return array of Mondrian keys for ``df_matches``."""

        if not self.keys:
            return np.array(["global"] * len(df_matches))

        df = df_matches.copy()
        for key in self.keys:
            if key == "OverroundBin":
                if "overround" not in df.columns:
                    raise KeyError("'overround' column required for OverroundBin")
                df[key] = pd.qcut(df["overround"], 10, labels=False, duplicates="drop")
            else:
                if key not in df.columns:
                    raise KeyError(f"Missing column '{key}' for Mondrian index")
        parts = [df[k].astype(str) for k in self.keys]
        combo = parts[0]
        for part in parts[1:]:
            combo = combo.str.cat(part, sep="|")
        return combo.to_numpy()

    def groups(self, df_matches: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(key, mask)`` pairs for each Mondrian group."""

        keys = self.make_keys(df_matches)
        for key in np.unique(keys):
            mask = keys == key
            yield key, mask


class ConformalIntervalPredictor:
    """Wrapper handling Mondrian partitioning and interval prediction."""

    def __init__(
        self,
        alpha: float,
        mondrian: MondrianIndexer | None = None,
        min_group_size: int = 100,
        width_cap: float = 0.6,
    ) -> None:
        self.alpha = alpha
        self.mondrian = mondrian or MondrianIndexer([])
        self.min_group_size = min_group_size
        self.width_cap = width_cap
        self.calibrators: Dict[str, VennAbersCalibrator] = {}
        self.global_calibrator: VennAbersCalibrator | None = None

    def fit(
        self, df_calib: pd.DataFrame, p_hat_cols: list[str], y_col: str
    ) -> "ConformalIntervalPredictor":
        """Fit calibrators for each Mondrian group."""

        p_hat = df_calib[p_hat_cols].to_numpy()
        y = df_calib[y_col].to_numpy()
        keys = self.mondrian.make_keys(df_calib)

        self.global_calibrator = VennAbersCalibrator(self.alpha).fit(p_hat, y)

        for key in np.unique(keys):
            mask = keys == key
            if mask.sum() < self.min_group_size:
                continue
            cal = VennAbersCalibrator(self.alpha).fit(p_hat[mask], y[mask])
            self.calibrators[key] = cal
        return self

    def _get_calibrator(self, key: str) -> VennAbersCalibrator:
        if self.global_calibrator is None:
            raise ValueError("Predictor not fitted")
        return self.calibrators.get(key, self.global_calibrator)

    def predict(
        self, df_test: pd.DataFrame, p_hat_cols: list[str]
    ) -> dict[str, np.ndarray]:
        """Predict probability intervals for ``df_test``."""

        p_hat = df_test[p_hat_cols].to_numpy()
        keys = self.mondrian.make_keys(df_test)
        n = len(df_test)
        n_classes = p_hat.shape[1]
        p_low = np.zeros((n, n_classes))
        p_high = np.zeros((n, n_classes))
        for key in np.unique(keys):
            mask = keys == key
            cal = self._get_calibrator(key)
            lo, hi = cal.predict_interval(p_hat[mask])
            width = np.clip(hi - lo, 0.0, self.width_cap)
            hi = np.clip(lo + width, 0.0, 1.0)
            p_low[mask] = lo
            p_high[mask] = hi
        return {"p_low": p_low, "p_high": p_high}
