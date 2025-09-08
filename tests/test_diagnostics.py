import numpy as np
import pandas as pd
from scipy import stats

from engine.eval.diagnostics import (
    ece_multiclass,
    reliability_table,
    pit_from_pmf,
    pit_diagnostics,
)
from engine.eval.regime import regime_report
from engine.eval.leakage import check_leakage


def test_ece_and_reliability():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    p_hat = np.zeros((200, 3))
    p_hat[np.arange(200), y] = 1.0
    ece = ece_multiclass(p_hat, y, n_bins=10)
    assert ece < 1e-6
    table = reliability_table(p_hat, y, n_bins=10)
    assert table["count"].sum() == len(y)


def test_pit_uniform():
    # simulate independent Poisson goals
    lam_h, lam_a = 1.2, 0.8
    max_g = 5
    gh = np.arange(max_g + 1)
    ga = np.arange(max_g + 1)
    pmf = np.outer(stats.poisson.pmf(gh, lam_h), stats.poisson.pmf(ga, lam_a))
    rng = np.random.default_rng(1)
    obs_h = rng.poisson(lam_h, size=500)
    obs_a = rng.poisson(lam_a, size=500)
    obs_h = np.clip(obs_h, 0, max_g)
    obs_a = np.clip(obs_a, 0, max_g)
    u = pit_from_pmf(obs_h, obs_a, pmf, rng)
    diag = pit_diagnostics(u)
    assert diag["ks_p"] > 0.05


def test_regime_report_clustering():
    df = pd.DataFrame({
        "overround_pre": [0.1]*50 + [0.2]*50,
        "avg_max_spread": [0.05]*50 + [0.1]*50,
        "market_disagreement": [0.02]*50 + [0.07]*50,
        "form_volatility_gap": [0.1]*50 + [0.3]*50,
        "y": [0]*50 + [1]*50,
    })
    preds = pd.DataFrame({
        "pH": [0.6]*100,
        "pD": [0.2]*100,
        "pA": [0.2]*100,
    })
    res = regime_report(df, preds, algo="kmeans", min_cluster_size=10, k_grid=[2,3])
    assert res.regime_map["regime_id"].nunique() >= 2


def test_leakage_warning():
    train = pd.DataFrame({
        "feat_close": [0.1, 0.2],
        "y": [0, 1],
        "event_time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
    })
    test = pd.DataFrame({
        "y": [1],
        "event_time": pd.to_datetime(["2020-01-03"]),
    })
    issues = check_leakage(train, test, {"market": ["feat_close"]})
    assert any(i.level == "warning" for i in issues)
