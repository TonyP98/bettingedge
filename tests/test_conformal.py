import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure package root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.risk.conformal import ConformalIntervalPredictor, MondrianIndexer
from engine.risk.thresholds import conformal_kelly
from engine.eval.metrics import empirical_coverage


def test_venn_abers_calibrator_coverage():
    np.random.seed(0)
    n_calib, n_test = 600, 300
    true_probs = np.random.dirichlet([3, 3, 3], size=n_calib + n_test)
    y = np.array([np.random.choice(3, p=p) for p in true_probs])
    p_hat = true_probs + np.random.normal(0, 0.05, size=true_probs.shape)
    p_hat = np.clip(p_hat, 0, 1)
    p_hat /= p_hat.sum(axis=1, keepdims=True)

    calib_df = pd.DataFrame(p_hat[:n_calib], columns=["pH", "pD", "pA"])
    calib_df["y"] = y[:n_calib]
    test_df = pd.DataFrame(p_hat[n_calib:], columns=["pH", "pD", "pA"])
    y_test = y[n_calib:]

    cip = ConformalIntervalPredictor(0.1, MondrianIndexer([]), min_group_size=50)
    cip.fit(calib_df, ["pH", "pD", "pA"], "y")
    intervals = cip.predict(test_df, ["pH", "pD", "pA"])
    true_p = true_probs[n_calib:]
    inside = []
    for i in range(len(test_df)):
        c = y_test[i]
        inside.append(
            intervals["p_low"][i, c] <= true_p[i, c] <= intervals["p_high"][i, c]
        )
    cov = np.mean(inside)
    assert abs(cov - 0.9) < 0.1


def test_conformal_interval_predictor_fallback():
    df = pd.DataFrame(
        {
            "Div": ["A", "B", "A", "B"],
            "overround": [0.1, 0.2, 0.15, 0.18],
            "pH": [0.3, 0.4, 0.5, 0.6],
            "pD": [0.3, 0.3, 0.2, 0.2],
            "pA": [0.4, 0.3, 0.3, 0.2],
            "y": [0, 1, 2, 0],
        }
    )
    cip = ConformalIntervalPredictor(0.1, MondrianIndexer(["OverroundBin"]), min_group_size=10)
    cip.fit(df, ["pH", "pD", "pA"], "y")
    preds = cip.predict(df, ["pH", "pD", "pA"])
    assert preds["p_low"].shape == (4, 3)


def test_conformal_kelly_cap_and_penalty():
    stake = conformal_kelly(0.6, 2.0, kelly_cap=0.15, variance_penalty=0.5, width=0.2)
    assert stake == 0.15
    stake2 = conformal_kelly(0.6, 2.0, kelly_cap=1.0, variance_penalty=0.0, width=0.2)
    stake3 = conformal_kelly(0.6, 2.0, kelly_cap=1.0, variance_penalty=1.0, width=0.2)
    assert stake2 > stake3
