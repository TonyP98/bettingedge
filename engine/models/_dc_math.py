import math
import numpy as np
from numpy.typing import NDArray

__all__ = ["dc_pmf_table", "probs_from_pmf"]

def _tau_correction(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles correction factor for a score (i, j).

    Parameters
    ----------
    i, j : int
        Goals scored by home and away team.
    lam_h, lam_a : float
        Poisson intensities for home and away team.
    rho : float
        Correlation correction parameter.
    """
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0

def dc_pmf_table(lambda_h: float, lambda_a: float, rho: float, max_goals: int = 6) -> NDArray[np.float64]:
    """Compute the Dixon-Coles joint PMF table.

    Parameters
    ----------
    lambda_h, lambda_a : float
        Expected goals for home and away teams.
    rho : float
        Dixon-Coles dependence parameter.
    max_goals : int, default 6
        Maximum number of goals for the grid.

    Returns
    -------
    np.ndarray
        Array of shape ``(max_goals+1, max_goals+1)`` with joint probabilities.
    """
    lam_h = float(np.clip(lambda_h, 1e-4, 10.0))
    lam_a = float(np.clip(lambda_a, 1e-4, 10.0))
    rho = float(np.clip(rho, -0.2, 0.2))
    grid = np.arange(max_goals + 1)
    factorial = np.vectorize(math.factorial)
    pois_h = np.exp(-lam_h) * np.power(lam_h, grid) / factorial(grid)
    pois_a = np.exp(-lam_a) * np.power(lam_a, grid) / factorial(grid)
    pmf = np.outer(pois_h, pois_a)
    # Apply Dixon-Coles tau correction
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            pmf[i, j] *= _tau_correction(i, j, lam_h, lam_a, rho)
    total = pmf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid PMF total")
    pmf /= total
    return pmf

def probs_from_pmf(pmf: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Aggregate probabilities from a joint PMF table.

    Parameters
    ----------
    pmf : np.ndarray
        Joint probability table produced by :func:`dc_pmf_table`.

    Returns
    -------
    tuple[float, float, float, float]
        Probabilities for home win, draw, away win and over 2.5 goals.
    """
    if pmf.ndim != 2:
        raise ValueError("pmf must be a 2D array")
    p_home = np.triu(pmf, k=1).sum()
    p_draw = np.trace(pmf)
    p_away = np.tril(pmf, k=-1).sum()
    # Over 2.5 goals
    goals = np.indices(pmf.shape).sum(axis=0)
    p_ou = pmf[goals > 2].sum()
    return float(p_home), float(p_draw), float(p_away), float(p_ou)
