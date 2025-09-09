from __future__ import annotations

import math

import numpy as np
import pandas as pd


MAX_GOALS = 8


def fit_team_rates(
    df_matches: pd.DataFrame, max_iter: int = 100, tol: float = 1e-6
) -> dict:
    """Estimate team attack/defense strengths and home advantage.

    Parameters
    ----------
    df_matches: DataFrame
        Must contain ``home``, ``away``, ``ft_home_goals`` and ``ft_away_goals``.
    max_iter: int
        Maximum number of Newton iterations.
    tol: float
        Convergence tolerance.
    """
    teams = sorted(set(df_matches["home"]) | set(df_matches["away"]))
    n = len(teams)
    team_idx = {team: i for i, team in enumerate(teams)}

    home_idx = df_matches["home"].map(team_idx).to_numpy()
    away_idx = df_matches["away"].map(team_idx).to_numpy()
    g_home = df_matches["ft_home_goals"].to_numpy(dtype=float)
    g_away = df_matches["ft_away_goals"].to_numpy(dtype=float)

    att = np.zeros(n, dtype=float)
    deff = np.zeros(n, dtype=float)
    home_adv = 0.0

    for _ in range(max_iter):
        lambda_home = np.exp(att[home_idx] + deff[away_idx] + home_adv)
        lambda_away = np.exp(att[away_idx] + deff[home_idx])

        grad_att = np.zeros(n, dtype=float)
        hess_att = np.zeros(n, dtype=float)
        grad_def = np.zeros(n, dtype=float)
        hess_def = np.zeros(n, dtype=float)

        np.add.at(grad_att, home_idx, g_home - lambda_home)
        np.add.at(grad_att, away_idx, g_away - lambda_away)
        np.add.at(hess_att, home_idx, -lambda_home)
        np.add.at(hess_att, away_idx, -lambda_away)

        np.add.at(grad_def, away_idx, g_home - lambda_home)
        np.add.at(grad_def, home_idx, g_away - lambda_away)
        np.add.at(hess_def, away_idx, -lambda_home)
        np.add.at(hess_def, home_idx, -lambda_away)

        grad_home = float(np.sum(g_home - lambda_home))
        hess_home = float(-np.sum(lambda_home))

        step_att = np.divide(
            grad_att, hess_att, out=np.zeros_like(grad_att), where=hess_att != 0
        )
        step_def = np.divide(
            grad_def, hess_def, out=np.zeros_like(grad_def), where=hess_def != 0
        )
        step_home = grad_home / hess_home if hess_home != 0 else 0.0

        att -= step_att
        deff -= step_def
        home_adv -= step_home

        att -= np.mean(att)
        deff -= np.mean(deff)

        if (
            np.max(np.abs(step_att)) < tol
            and np.max(np.abs(step_def)) < tol
            and abs(step_home) < tol
        ):
            break

    return {
        "attack": {team: float(att[team_idx[team]]) for team in teams},
        "defense": {team: float(deff[team_idx[team]]) for team in teams},
        "home_adv": float(home_adv),
    }


def _poisson_pmf(lmbda: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(0, max_goals + 1)
    fact = np.array([math.factorial(i) for i in goals], dtype=float)
    return np.exp(-lmbda) * np.power(lmbda, goals) / fact


def predict_match_probs(df_matches: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Predict 1X2 probabilities for each match using Poisson convolution."""
    teams_att = rates.get("attack", {})
    teams_def = rates.get("defense", {})
    home_adv = rates.get("home_adv", 0.0)

    rows = []
    for _, row in df_matches.iterrows():
        att_home = teams_att.get(row["home"], 0.0)
        def_home = teams_def.get(row["home"], 0.0)
        att_away = teams_att.get(row["away"], 0.0)
        def_away = teams_def.get(row["away"], 0.0)

        lambda_home = math.exp(att_home + def_away + home_adv)
        lambda_away = math.exp(att_away + def_home)

        p_home = _poisson_pmf(lambda_home)
        p_away = _poisson_pmf(lambda_away)
        mat = np.outer(p_home, p_away)
        total = mat.sum()
        p1 = np.triu(mat, 1).sum() / total
        px = np.trace(mat) / total
        p2 = np.tril(mat, -1).sum() / total
        rows.append(
            {
                "p1": p1,
                "px": px,
                "p2": p2,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
            }
        )

    return pd.DataFrame(rows, index=df_matches.index)
