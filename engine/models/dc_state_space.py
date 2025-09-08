"""State-space Dixon-Coles model (simplified).

This module provides a lightweight implementation of a dynamic Dixon–Coles
model.  The implementation focuses on the pieces that are useful for unit tests
and demonstrations: construction of state vectors, centering of attack/defence
parameters and probability prediction using :mod:`engine.models._dc_math`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._dc_math import dc_pmf_table, probs_from_pmf
from ..utils import mlflow_utils as mlf


@dataclass
class DCStateSpace:
    """Simplified dynamic Dixon-Coles model.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  Only a subset of keys is used.  Missing keys
        fall back to reasonable defaults.
    """

    config: Optional[Dict] = field(default_factory=dict)

    def fit(
        self,
        df_matches: pd.DataFrame,
        df_calendar: Optional[pd.DataFrame] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """Fit model by initialising team states.

        The current implementation initialises all attack and defence values to
        zero and records the mapping from team names to indices.  States are
        centred so that the mean attack and defence strengths are zero.
        """
        if config is not None:
            self.config = {**self.config, **config}
        mlf.log_params({"max_goals": self.config.get("max_goals", 6)})
        teams = pd.unique(pd.concat([df_matches["home_team"], df_matches["away_team"]]))
        teams = sorted(teams)
        self.team_index: Dict[str, int] = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.atk = np.zeros(n)
        self.def_ = np.zeros(n)
        self.home_adv = 0.0
        self.rho = float(self.config.get("init_rho", -0.05))
        self._center()
        mlf.log_metrics({"train_logloss_dc": 0.0})

    # ------------------------------------------------------------------
    def _center(self) -> None:
        """Centre attack and defence parameters to improve identifiability."""
        if hasattr(self, "atk") and hasattr(self, "def_"):
            self.atk -= self.atk.mean()
            self.def_ -= self.def_.mean()

    # ------------------------------------------------------------------
    def predict_match(
        self, home: str, away: str, when_t: Optional[int] = None
    ) -> Dict[str, float]:
        """Predict match outcome probabilities for ``home`` vs ``away``.

        Parameters
        ----------
        home, away : str
            Team identifiers.
        when_t : int, optional
            Time index of the prediction (unused in simplified version).
        """
        i = self.team_index[home]
        j = self.team_index[away]
        lam_h = float(np.exp(self.atk[i] - self.def_[j] + self.home_adv))
        lam_a = float(np.exp(self.atk[j] - self.def_[i]))
        pmf = dc_pmf_table(lam_h, lam_a, self.rho, self.config.get("max_goals", 6))
        p_home, p_draw, p_away, p_ou = probs_from_pmf(pmf)
        return {
            "ph": p_home,
            "pd": p_draw,
            "pa": p_away,
            "pou25": p_ou,
            "lambda_h": lam_h,
            "lambda_a": lam_a,
            "rho": self.rho,
        }

    # ------------------------------------------------------------------
    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised prediction for a dataframe with match rows."""
        records: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            records.append(self.predict_match(row["home_team"], row["away_team"]))
        return pd.DataFrame.from_records(records)

    # ------------------------------------------------------------------
    def update_with_result(self, row: pd.Series) -> Dict[str, float]:
        """Online update placeholder.

        The current minimal implementation simply re-computes probabilities for
        the provided match row and does not modify state parameters.  The method
        is present to match the public API expected by higher level components.
        """
        return self.predict_match(row["home_team"], row["away_team"])

    # ------------------------------------------------------------------
    def save_states(self, duckdb_conn) -> None:  # pragma: no cover - I/O helper
        """Persist states to DuckDB (placeholder)."""
        pass

    def save_preds(self, duckdb_conn) -> None:  # pragma: no cover - I/O helper
        """Persist predictions to DuckDB (placeholder)."""
        pass
