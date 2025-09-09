from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..model.poisson import MAX_GOALS, _poisson_pmf


@dataclass
class MarketSpec:
    """Specification for a betting market."""

    name: str
    targets: Sequence[str]
    odds_cols: Mapping[str, str]
    settle_fn: Callable[[int, int, str, float, float | None], float]
    prob_fn: Callable[
        [pd.Series, Mapping[str, float], Mapping[str, float]], Mapping
    ]


def prob_1x2(
    _df_row: pd.Series, model_probs: Mapping[str, float], _extra: Mapping[str, float]
) -> Mapping[str, float]:
    """Return model probabilities for the 1X2 market."""

    return {"1": model_probs["p1"], "x": model_probs["px"], "2": model_probs["p2"]}


def prob_ou25(
    _df_row: pd.Series,
    _model_probs: Mapping[str, float],
    model_poisson_extra: Mapping[str, float],
) -> Mapping[str, float]:
    """Probability of total goals being over/under 2.5 using a Poisson model."""

    lam_home = model_poisson_extra.get("lambda_home")
    lam_away = model_poisson_extra.get("lambda_away")
    if lam_home is None or lam_away is None:
        return {}
    p_total = _poisson_pmf(lam_home + lam_away, MAX_GOALS * 2)
    p_under = float(np.sum(p_total[:3]))
    return {"over25": 1 - p_under, "under25": p_under}


def prob_dnb(
    _df_row: pd.Series, model_probs: Mapping[str, float], _extra: Mapping[str, float]
) -> Mapping[str, float]:
    """Derive draw-no-bet probabilities from 1X2 model probs."""

    p1 = model_probs["p1"]
    p2 = model_probs["p2"]
    denom = p1 + p2
    if denom == 0:
        return {}
    return {"home": p1 / denom, "away": p2 / denom}


def prob_dc(
    _df_row: pd.Series, model_probs: Mapping[str, float], _extra: Mapping[str, float]
) -> Mapping[str, float]:
    """Compute double chance probabilities from 1X2 model probs."""

    p1 = model_probs["p1"]
    px = model_probs["px"]
    p2 = model_probs["p2"]
    return {"1x": p1 + px, "12": p1 + p2, "x2": px + p2}


def _score_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    p_home = _poisson_pmf(lambda_home, MAX_GOALS)
    p_away = _poisson_pmf(lambda_away, MAX_GOALS)
    mat = np.outer(p_home, p_away)
    return mat / mat.sum()


def _diff_probs(mat: np.ndarray) -> Mapping[int, float]:
    diff_probs: dict[int, float] = {}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            diff = i - j
            diff_probs[diff] = diff_probs.get(diff, 0.0) + float(mat[i, j])
    return diff_probs


def _ah_selection_probs(
    diff_probs: Mapping[int, float], line: float, invert: bool = False
) -> Mapping[str, float]:
    out = {
        "win": 0.0,
        "half_win": 0.0,
        "void": 0.0,
        "half_loss": 0.0,
        "loss": 0.0,
    }
    for diff, prob in diff_probs.items():
        d = -diff if invert else diff
        if (line * 2) % 1 == 0:  # multiples of 0.5
            if d > line:
                out["win"] += prob
            elif d == line:
                out["void"] += prob
            else:
                out["loss"] += prob
        else:  # quarter lines
            lower = line - 0.25
            upper = line + 0.25
            r1 = "win" if d > lower else "void" if d == lower else "loss"
            r2 = "win" if d > upper else "void" if d == upper else "loss"
            if r1 == "win" and r2 == "win":
                out["win"] += prob
            elif "win" in (r1, r2) and "void" in (r1, r2):
                out["half_win"] += prob
            elif r1 == "void" and r2 == "void":
                out["void"] += prob
            elif "loss" in (r1, r2) and "void" in (r1, r2):
                out["half_loss"] += prob
            else:
                out["loss"] += prob
    return out


def prob_ah(
    df_row: pd.Series,
    _model_probs: Mapping[str, float],
    model_poisson_extra: Mapping[str, float],
) -> Mapping[str, Mapping[str, float]]:
    """Asian handicap probabilities for home and away selections."""

    line = df_row.get("ah_line")
    if pd.isna(line):
        return {}
    lam_home = model_poisson_extra.get("lambda_home")
    lam_away = model_poisson_extra.get("lambda_away")
    if lam_home is None or lam_away is None:
        return {}
    mat = _score_matrix(lam_home, lam_away)
    diff_probs = _diff_probs(mat)
    home = _ah_selection_probs(diff_probs, line)
    away = _ah_selection_probs(diff_probs, -line, invert=True)
    return {"home": home, "away": away}


def settle_1x2(
    ft_home_goals: int, ft_away_goals: int, selection: str, odds: float, _: float | None = None
) -> float:
    """Settle a 1X2 bet."""

    if selection == "1":
        won = ft_home_goals > ft_away_goals
    elif selection == "x":
        won = ft_home_goals == ft_away_goals
    else:
        won = ft_home_goals < ft_away_goals
    return (odds - 1) if won else -1.0


def settle_ou25(
    ft_home_goals: int, ft_away_goals: int, selection: str, odds: float, _: float | None = None
) -> float:
    """Settle an over/under 2.5 bet."""

    total = ft_home_goals + ft_away_goals
    if selection == "over25":
        won = total >= 3
    else:
        won = total <= 2
    return (odds - 1) if won else -1.0


def settle_dnb(
    ft_home_goals: int, ft_away_goals: int, selection: str, odds: float, _: float | None = None
) -> float:
    """Settle a draw-no-bet wager."""

    if ft_home_goals == ft_away_goals:
        return 0.0
    if selection == "home":
        won = ft_home_goals > ft_away_goals
    else:
        won = ft_away_goals > ft_home_goals
    return (odds - 1) if won else -1.0


def settle_dc(
    ft_home_goals: int, ft_away_goals: int, selection: str, odds: float, _: float | None = None
) -> float:
    """Settle a double chance bet treated as a two-way market."""

    if selection == "1x":
        won = ft_home_goals >= ft_away_goals
    elif selection == "12":
        won = ft_home_goals != ft_away_goals
    else:  # x2
        won = ft_home_goals <= ft_away_goals
    return (odds - 1) if won else -1.0


def _single_ah_payoff(diff: int, line: float, odds: float) -> float:
    if diff > line:
        return odds - 1
    if diff == line:
        return 0.0
    return -1.0


def settle_ah(
    ft_home_goals: int, ft_away_goals: int, selection: str, odds: float, ah_line: float
) -> float:
    """Settle an Asian handicap bet supporting quarter lines."""

    diff = ft_home_goals - ft_away_goals
    line = ah_line
    if selection == "away":
        diff = -diff
        line = -line
    if (line * 2) % 1 == 0:  # integer or half line
        return _single_ah_payoff(diff, line, odds)
    lower = line - 0.25
    upper = line + 0.25
    return 0.5 * (
        _single_ah_payoff(diff, lower, odds) + _single_ah_payoff(diff, upper, odds)
    )


spec_1x2 = MarketSpec(
    name="1x2",
    targets=("1", "x", "2"),
    odds_cols={"1": "odds_1", "x": "odds_x", "2": "odds_2"},
    settle_fn=settle_1x2,
    prob_fn=prob_1x2,
)


spec_ou25 = MarketSpec(
    name="ou25",
    targets=("over25", "under25"),
    odds_cols={"over25": "odds_over25", "under25": "odds_under25"},
    settle_fn=settle_ou25,
    prob_fn=prob_ou25,
)


spec_dnb = MarketSpec(
    name="dnb",
    targets=("home", "away"),
    odds_cols={"home": "odds_dnb_home", "away": "odds_dnb_away"},
    settle_fn=settle_dnb,
    prob_fn=prob_dnb,
)


spec_dc = MarketSpec(
    name="dc",
    targets=("1x", "12", "x2"),
    odds_cols={"1x": "odds_1x", "12": "odds_12", "x2": "odds_x2"},
    settle_fn=settle_dc,
    prob_fn=prob_dc,
)


spec_ah = MarketSpec(
    name="ah",
    targets=("home", "away"),
    odds_cols={"home": "odds_ah_home", "away": "odds_ah_away"},
    settle_fn=settle_ah,
    prob_fn=prob_ah,
)


MARKETS = {
    "1x2": spec_1x2,
    "ou25": spec_ou25,
    "dnb": spec_dnb,
    "dc": spec_dc,
    "ah": spec_ah,
}

