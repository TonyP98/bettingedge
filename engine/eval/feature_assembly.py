"""Utilities to build the design matrix for the ensemble meta learner."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

__all__ = ["assemble_ensemble_features", "assemble_explain_matrix"]


def assemble_ensemble_features(
    df: pd.DataFrame, *, include_close: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """Assemble features and target for the meta learner.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe containing probabilities and auxiliary features.
    include_close : bool, default False
        Whether to include market closing probabilities. These are only
        available at test time and must not be used for training.
    """
    base_cols = [
        "pH_DC",
        "pD_DC",
        "pA_DC",
        "pOU_DC",
        "pH_BP",
        "pD_BP",
        "pA_BP",
        "pOU_BP",
        "pH_MKT_pre",
        "pD_MKT_pre",
        "pA_MKT_pre",
    ]
    if include_close:
        for col in ["pH_MKT_close", "pD_MKT_close", "pA_MKT_close"]:
            if col in df.columns:
                base_cols.append(col)
    form_cols = [c for c in df.columns if c.startswith("form_")]
    micro_cols = [c for c in df.columns if c.startswith("micro_")]
    cols = base_cols + form_cols + micro_cols
    X = df[cols].copy()
    y = df["y_wdl"].copy()
    return X, y


def assemble_explain_matrix(
    df: pd.DataFrame, *, include_close: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble features, target and identifiers for explainability tasks.

    This helper is a thin wrapper around :func:`assemble_ensemble_features` that
    also returns the ``match_id`` column (if present) used to index SHAP values
    and rule-based playbooks.
    """

    X, y = assemble_ensemble_features(df, include_close=include_close)
    match_ids = df["match_id"] if "match_id" in df.columns else pd.Series(
        range(len(df)), name="match_id"
    )
    return X, y, match_ids
