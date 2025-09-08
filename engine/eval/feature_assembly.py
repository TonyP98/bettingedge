"""Utilities to build the design matrix for the ensemble meta learner."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

__all__ = ["assemble_ensemble_features"]


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
