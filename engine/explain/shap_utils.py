"""SHAP utility functions for model explainability."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["shap_global", "shap_local"]


def _get_shap_values(model, X_bg: pd.DataFrame, X_eval: pd.DataFrame):
    """Compute shap values using TreeExplainer.

    Parameters
    ----------
    model : fitted tree-based model with ``predict``/``predict_proba`` methods.
    X_bg : pd.DataFrame
        Background dataset used to initialise the explainer.
    X_eval : pd.DataFrame
        Evaluation dataset for which SHAP values are required.
    """
    try:
        import shap
    except ImportError as e:
        raise RuntimeError("shap non è installato: pip install .[explain]") from e

    explainer = shap.TreeExplainer(
        model,
        X_bg,
        feature_perturbation="interventional",
        model_output="probability",
    )
    shap_values = explainer.shap_values(X_eval)
    base_value = explainer.expected_value
    # For binary classification shap returns list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        if isinstance(base_value, Iterable):
            base_value = base_value[1]
    return shap_values, base_value


def shap_global(
    model,
    X_bg: pd.DataFrame,
    X_eval: pd.DataFrame,
    feature_names: list[str],
    fold: int,
) -> pd.DataFrame:
    """Compute global feature importance using mean absolute SHAP values.

    Returns a dataframe with columns ``feature`` and ``mean_abs`` along with the
    ``fold`` identifier and creation timestamp.
    """
    shap_values, _ = _get_shap_values(model, X_bg, X_eval)
    df = (
        pd.DataFrame(shap_values, columns=feature_names)
        .abs()
        .mean()
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_abs"})
    )
    df["fold"] = fold
    df["created_at"] = datetime.utcnow()
    return df


def shap_local(
    model,
    X_bg: pd.DataFrame,
    X_eval: pd.DataFrame,
    match_ids: Iterable[str],
    feature_names: list[str],
    fold: int,
    topk: int = 8,
) -> pd.DataFrame:
    """Compute local SHAP values for individual matches.

    Parameters
    ----------
    model : fitted model
    X_bg : pd.DataFrame
        Background dataset for the explainer.
    X_eval : pd.DataFrame
        Rows for which explanations are desired.
    match_ids : iterable
        Identifiers for the rows in ``X_eval``.
    feature_names : list[str]
        Names of the features corresponding to the columns of ``X_eval``.
    fold : int
        Fold identifier.
    topk : int, default ``8``
        Number of features with highest absolute SHAP value to retain per row.
    """
    shap_values, base_value = _get_shap_values(model, X_bg, X_eval)
    df_vals = pd.DataFrame(shap_values, columns=feature_names)
    records: list[dict] = []
    for i, mid in enumerate(match_ids):
        row = df_vals.iloc[i]
        top_features = row.abs().nlargest(min(topk, len(row)))
        for feat in top_features.index:
            records.append(
                {
                    "match_id": mid,
                    "fold": fold,
                    "feature": feat,
                    "shap_value": row[feat],
                    "base_value": base_value,
                    "created_at": datetime.utcnow(),
                }
            )
    return pd.DataFrame.from_records(records)
