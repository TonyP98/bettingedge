"""Utilities to build and encode context features for bandit agents."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def build_context_row(match_row, aux_features: Dict[str, Any]) -> Dict[str, Any]:
    """Build a context dictionary from a match row and extra features.

    The function simply merges the provided ``match_row`` (either a
    ``Series`` or mapping) with ``aux_features`` into a flat dictionary.
    Missing values are kept as ``NaN`` and handled during encoding.
    """

    ctx: Dict[str, Any] = {}
    if hasattr(match_row, "to_dict"):
        ctx.update(match_row.to_dict())
    else:
        ctx.update(dict(match_row))
    ctx.update(aux_features)
    return ctx


def encode_context(df_ctx: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Encode context rows into a numerical design matrix.

    Numeric columns are imputed with their median while categorical
    columns are one-hot encoded using pandas ``get_dummies``.
    """

    df = df_ctx.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    X = df.to_numpy(dtype=float)
    meta = {"columns": list(df.columns)}
    return X, meta
