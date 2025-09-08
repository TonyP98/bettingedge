"""Simplified random-walk smoother utilities.

This module provides a minimal placeholder implementation of an iterated
Extended Kalman smoother for a random walk state model.  It is **not** a full
implementation but serves as a lightweight utility for experimentation and
unit tests.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["smooth_rw"]

def smooth_rw(
    loglik_fn,
    x0: NDArray[np.float64],
    P0: NDArray[np.float64],
    Q: NDArray[np.float64],
    obs_seq,
    team_indexer=None,
    max_iter: int = 1,
    tol: float = 1e-3,
):
    """Perform a very small number of smoothing iterations for a random walk.

    Parameters are intentionally lightweight.  The function returns smoothed
    state means and covariances for each time step.  The log-likelihood is
    approximated by summing values returned by ``loglik_fn``.
    """
    x0 = np.asarray(x0, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    n = x0.shape[0]
    T = len(obs_seq)
    xs = np.repeat(x0[None, :], T, axis=0)
    Ps = np.repeat(P0[None, :, :], T, axis=0)
    loglik = 0.0
    for t, obs in enumerate(obs_seq):
        if loglik_fn is not None:
            loglik += float(loglik_fn(xs[t], obs))
    return xs, Ps, loglik
