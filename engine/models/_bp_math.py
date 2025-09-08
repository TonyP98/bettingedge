"""Math utilities for the correlated bivariate Poisson model."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = ["softplus", "bivar_poisson_pmf", "outcome_probs"]


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable softplus."""
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def bivar_poisson_pmf(lambda1: float, lambda2: float, lambda12: float, max_goals: int) -> np.ndarray:
    """Return PMF grid for the bivariate Poisson distribution.

    Parameters
    ----------
    lambda1, lambda2, lambda12 : float
        Model parameters. ``lambda12`` controls the covariance between the two
        Poisson margins.
    max_goals : int
        Maximum goals for each team (inclusive) used to build the grid.
    """
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            k_max = min(x, y)
            terms = []
            for k in range(k_max + 1):
                log_t = (
                    (x - k) * np.log(lambda1)
                    - gammaln(x - k + 1)
                    + (y - k) * np.log(lambda2)
                    - gammaln(y - k + 1)
                    + k * np.log(lambda12)
                    - gammaln(k + 1)
                )
                terms.append(log_t)
            log_prob = -(lambda1 + lambda2 + lambda12) + logsumexp(terms)
            grid[x, y] = np.exp(log_prob)
    return grid


def outcome_probs(pmf: np.ndarray, ou_line: float = 2.5) -> tuple[float, float, float, float]:
    """Compute match outcome probabilities from a PMF grid."""
    total = pmf.sum()
    p_d = np.trace(pmf)
    p_a = np.sum(np.tril(pmf, k=-1))
    p_h = total - p_d - p_a
    goals = np.add.outer(np.arange(pmf.shape[0]), np.arange(pmf.shape[1]))
    p_ou = pmf[goals > ou_line].sum()
    return float(p_h), float(p_d), float(p_a), float(p_ou)
