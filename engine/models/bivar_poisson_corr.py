"""Bivariate Poisson model with a correlation term.

This is a light-weight implementation sufficient for unit testing. The model
fits a simple GLM for the shared component ``lambda12`` while ``lambda1`` and
``lambda2`` are computed from pre-supplied attack/defence strengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._bp_math import bivar_poisson_pmf, outcome_probs, softplus
from ..utils import mlflow_utils as mlf


@dataclass
class BivarPoisson:
    """Correlated bivariate Poisson model."""

    config: Dict | None = None
    coef_: np.ndarray | None = None

    def fit(
        self, df_matches: pd.DataFrame, config: Dict | None = None
    ) -> "BivarPoisson":
        """Fit the model on a dataframe of matches."""
        self.config = {"max_goals": 6, "reg_lambda12": 1e-3}
        if config:
            self.config.update(config)
        mlf.log_params(self.config)

        z_cols = [c for c in df_matches.columns if c.startswith("z")]
        Z = df_matches[z_cols].to_numpy()
        y1 = df_matches["home_goals"].to_numpy()
        y2 = df_matches["away_goals"].to_numpy()

        # lambdas 1 and 2 are derived from pre-computed attack/defence ratings
        l1 = np.exp(
            df_matches["atk_home"] - df_matches["def_away"] + df_matches["home_adv"]
        )
        l2 = np.exp(df_matches["atk_away"] - df_matches["def_home"])

        def nll(w: np.ndarray) -> float:
            lam12 = softplus(Z @ w)
            ll = 0.0
            for lam1, lam2, lamc, g1, g2 in zip(l1, l2, lam12, y1, y2):
                pmf = bivar_poisson_pmf(lam1, lam2, lamc, self.config["max_goals"])
                # ensure grid covers observed goals
                lam12_clip = max(lamc, 1e-9)
                pmf_val = 0.0
                if g1 <= self.config["max_goals"] and g2 <= self.config["max_goals"]:
                    pmf_val = pmf[g1, g2]
                else:  # fall back to independent Poisson outside grid
                    pmf_val = (
                        np.exp(-(lam1 + lam2 + lam12_clip))
                        * lam1**g1
                        / np.math.factorial(g1)
                        * lam2**g2
                        / np.math.factorial(g2)
                    )
                ll += np.log(pmf_val + 1e-12)
            reg = 0.5 * self.config["reg_lambda12"] * np.sum(w**2)
            return -(ll - reg)

        w0 = np.zeros(Z.shape[1])
        res = minimize(nll, w0, method="BFGS")
        self.coef_ = res.x
        mlf.log_metrics({"nll_final": float(res.fun)})
        return self

    def _predict_row(self, row: pd.Series) -> Dict:
        z_cols = [c for c in row.index if c.startswith("z")]
        z = row[z_cols].to_numpy()
        lam1 = float(np.exp(row["atk_home"] - row["def_away"] + row["home_adv"]))
        lam2 = float(np.exp(row["atk_away"] - row["def_home"]))
        lam12 = float(softplus(z @ self.coef_))
        pmf = bivar_poisson_pmf(lam1, lam2, lam12, self.config["max_goals"])
        p_h, p_d, p_a, p_ou = outcome_probs(pmf)
        return {
            "pH_BP": p_h,
            "pD_BP": p_d,
            "pA_BP": p_a,
            "pOU_BP": p_ou,
            "lambda_h": lam1,
            "lambda_a": lam2,
            "lambda12": lam12,
        }

    def predict_match(self, row: pd.Series) -> Dict:
        return self._predict_row(row)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = [self._predict_row(r) for _, r in df.iterrows()]
        return pd.DataFrame(preds)
