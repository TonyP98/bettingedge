import numpy as np

from engine.models._dc_math import dc_pmf_table, probs_from_pmf


def test_pmf_sums_to_one():
    pmf = dc_pmf_table(1.2, 0.8, 0.1, max_goals=6)
    assert np.isclose(pmf.sum(), 1.0, atol=1e-6)


def test_probabilities_from_pmf():
    pmf = dc_pmf_table(1.0, 1.0, 0.0, max_goals=6)
    ph, pd, pa, pou = probs_from_pmf(pmf)
    assert np.isclose(ph + pd + pa, 1.0, atol=1e-6)
    assert 0 <= pou <= 1
