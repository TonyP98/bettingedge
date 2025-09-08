import numpy as np
import pandas as pd
import pytest

from engine.eval.feature_assembly import assemble_ensemble_features
from engine.models.meta_learner import MetaEnsemble


def _generate_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "pH_DC": rng.random(n),
            "pD_DC": rng.random(n),
            "pA_DC": rng.random(n),
            "pOU_DC": rng.random(n),
            "pH_BP": rng.random(n),
            "pD_BP": rng.random(n),
            "pA_BP": rng.random(n),
            "pOU_BP": rng.random(n),
            "pH_MKT_pre": rng.random(n),
            "pD_MKT_pre": rng.random(n),
            "pA_MKT_pre": rng.random(n),
            "pH_MKT_close": rng.random(n),
            "pD_MKT_close": rng.random(n),
            "pA_MKT_close": rng.random(n),
            "form_diff": rng.normal(size=n),
            "micro_overround": rng.random(n),
        }
    )
    probs = df[["pH_DC", "pD_DC", "pA_DC"]].to_numpy()
    probs /= probs.sum(axis=1, keepdims=True)
    df["pH_DC"], df["pD_DC"], df["pA_DC"] = probs.T
    df["y_wdl"] = rng.choice([0, 1, 2], size=n)
    return df


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    confidences = proba[np.arange(len(y_true)), y_true]
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.any():
            acc = (y_true[mask] == proba[mask].argmax(axis=1)).mean()
            conf = confidences[mask].mean()
            ece_val += abs(acc - conf) * mask.mean()
    return ece_val


def test_anti_leak_enforced():
    df = _generate_df(20)
    X, y = assemble_ensemble_features(df, include_close=False)
    assert all("close" not in c for c in X.columns)
    X_bad = X.copy()
    X_bad["pH_MKT_close"] = 0.1
    model = MetaEnsemble()
    with pytest.raises(ValueError):
        model.fit(X_bad, y)


def test_predict_proba_and_calibration():
    df = _generate_df(120)
    train = df.iloc[:80]
    test = df.iloc[80:]
    X_train, y_train = assemble_ensemble_features(train, include_close=False)
    X_test, y_test = assemble_ensemble_features(test, include_close=False)
    model = MetaEnsemble()
    model.fit(X_train, y_train, calibrate=True)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    raw = model.model.predict_proba(X_test)
    assert ece(y_test.to_numpy(), proba) <= ece(y_test.to_numpy(), raw) + 1e-6
