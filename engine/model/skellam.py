from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from .common import ScoreGrid


def _decay_weights(dates: pd.Series, half_life_days: int) -> np.ndarray:
    max_date = dates.max()
    dt = (max_date - dates).dt.days.to_numpy()
    return 2 ** (-dt / half_life_days)


def fit_skellam(
    df_train: pd.DataFrame, *, half_life_days: int = 120, l2: float = 1e-3
) -> dict:
    """Fit a Skellam (independent Poisson) model."""

    df = df_train.dropna(subset=["ft_home_goals", "ft_away_goals"]).copy()
    if df.empty:
        raise ValueError("Training dataframe is empty after dropping missing scores")

    teams = sorted(set(df["home"]).union(df["away"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = df["home"].map(team_to_idx).to_numpy()
    away_idx = df["away"].map(team_to_idx).to_numpy()
    home_goals = df["ft_home_goals"].to_numpy(dtype=float)
    away_goals = df["ft_away_goals"].to_numpy(dtype=float)
    weights = _decay_weights(df["date"], half_life_days)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        a = np.concatenate([[0.0], params[: n - 1]])
        d = np.concatenate([[0.0], params[n - 1 : 2 * (n - 1)]])
        home_adv = params[-1]
        return a, d, home_adv

    def nll(params: np.ndarray) -> float:
        a, d, home_adv = unpack(params)
        lambda_h = np.exp(home_adv + a[home_idx] - d[away_idx])
        lambda_a = np.exp(a[away_idx] - d[home_idx])
        log_p_h = home_goals * np.log(lambda_h) - lambda_h - gammaln(home_goals + 1)
        log_p_a = away_goals * np.log(lambda_a) - lambda_a - gammaln(away_goals + 1)
        ll = log_p_h + log_p_a
        return -np.sum(weights * ll) + l2 * np.sum(params**2)

    x0 = np.zeros(2 * (n - 1) + 1)
    bounds = [(-2.0, 2.0)] * (2 * (n - 1)) + [(-2.0, 2.0)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    attack, defense, home_adv = unpack(res.x)
    return {
        "attack": {team: attack[i] for i, team in enumerate(teams)},
        "defense": {team: defense[i] for i, team in enumerate(teams)},
        "home_adv": float(home_adv),
    }


def _poisson_matrix(lambda_h: float, lambda_a: float, max_goals: int) -> np.ndarray:
    i = np.arange(max_goals + 1)
    j = np.arange(max_goals + 1)
    p = poisson.pmf(i[:, None], lambda_h) * poisson.pmf(j[None, :], lambda_a)
    p /= p.sum()
    return p


def predict_skellam_grid(
    df_all: pd.DataFrame, params: dict, max_goals: int = 10
) -> ScoreGrid:
    """Generate score grids assuming independent Poisson goals."""

    df = df_all.copy()
    if "match_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "match_id"})
    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        atk = params["attack"].get(row.home, 0.0)
        dfn = params["defense"].get(row.away, 0.0)
        atk_a = params["attack"].get(row.away, 0.0)
        dfn_h = params["defense"].get(row.home, 0.0)
        lambda_h = np.exp(params["home_adv"] + atk - dfn)
        lambda_a = np.exp(atk_a - dfn_h)
        if not np.isfinite(lambda_h) or not np.isfinite(lambda_a):
            raise ValueError("Non-finite lambda encountered")
        grid = _poisson_matrix(lambda_h, lambda_a, max_goals)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                records.append(
                    {
                        "match_id": row.match_id,
                        "date": row.date,
                        "home": row.home,
                        "away": row.away,
                        "i": i,
                        "j": j,
                        "p_ij": grid[i, j],
                    }
                )
    grid_df = pd.DataFrame.from_records(records)
    return ScoreGrid(grid_df)
