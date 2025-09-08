"""Stacked meta learner with optional isotonic calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression


@dataclass
class MetaEnsemble:
    """LightGBM based stacked ensemble with calibration."""

    params: Dict[str, Any] | None = None
    calibrators: list[IsotonicRegression] | None = None
    model: LGBMClassifier | None = None

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, *, calibrate: bool = True
    ) -> "MetaEnsemble":
        if any("close" in c for c in X_train.columns):
            raise ValueError("Training data contains close-market features")
        params = {"num_class": 3, "objective": "multiclass"}
        if self.params:
            params.update(self.params)
        self.model = LGBMClassifier(**params)
        self.model.fit(X_train, y_train)
        if calibrate:
            raw = self.model.predict_proba(X_train)
            self.calibrators = []
            for k in range(raw.shape[1]):
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(raw[:, k], (y_train == k).astype(int))
                self.calibrators.append(ir)
        return self

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X_test)
        if self.calibrators:
            calibrated = np.column_stack(
                [cal.predict(raw[:, k]) for k, cal in enumerate(self.calibrators)]
            )
            calibrated = np.clip(calibrated, 1e-12, 1 - 1e-12)
            calibrated /= calibrated.sum(axis=1, keepdims=True)
            return calibrated
        return raw

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "calibrators": self.calibrators}, path)

    @staticmethod
    def load(path: str) -> "MetaEnsemble":
        data = joblib.load(path)
        obj = MetaEnsemble()
        obj.model = data["model"]
        obj.calibrators = data["calibrators"]
        return obj
