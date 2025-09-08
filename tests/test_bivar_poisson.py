import numpy as np
import pandas as pd
import pytest

from engine.models._bp_math import bivar_poisson_pmf, outcome_probs
from engine.models.bivar_poisson_corr import BivarPoisson


def test_pmf_sums_to_one():
    pmf = bivar_poisson_pmf(1.0, 1.2, 0.3, max_goals=10)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)
    p_h, p_d, p_a, _ = outcome_probs(pmf)
    assert p_h + p_d + p_a == pytest.approx(1.0, rel=1e-6)


def test_fit_and_predict():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "atk_home": rng.normal(1.0, 0.1, 8),
            "def_home": rng.normal(0.5, 0.1, 8),
            "atk_away": rng.normal(0.8, 0.1, 8),
            "def_away": rng.normal(0.6, 0.1, 8),
            "home_adv": 0.1,
            "z1": rng.normal(size=8),
            "z2": rng.normal(size=8),
            "z3": rng.normal(size=8),
            "home_goals": rng.poisson(1.2, 8),
            "away_goals": rng.poisson(1.0, 8),
        }
    )
    model = BivarPoisson().fit(df, {"max_goals": 10})
    preds = model.predict_df(df)
    assert np.allclose(preds[["pH_BP", "pD_BP", "pA_BP"]].sum(axis=1), 1.0, atol=1e-3)
    assert (preds["lambda12"] >= 0).all()
