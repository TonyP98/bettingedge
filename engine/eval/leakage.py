"""Automatic checks for data leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class LeakageIssue:
    level: str
    message: str
    locations: List[str]


def check_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: Dict[str, Iterable[str]],
) -> List[LeakageIssue]:
    """Run basic leakage checks and return list of issues."""
    issues: List[LeakageIssue] = []

    close_cols = [c for c in train_df.columns if c.endswith("_close")]
    if close_cols:
        issues.append(
            LeakageIssue(
                level="warning",
                message="future-like '_close' features found in training data",
                locations=close_cols,
            )
        )

    if "event_time" in train_df.columns and "event_time" in test_df.columns:
        if pd.to_datetime(test_df["event_time"].min()) <= pd.to_datetime(train_df["event_time"].max()):
            issues.append(
                LeakageIssue(
                    level="error",
                    message="test set not strictly after training set",
                    locations=["event_time"],
                )
            )

    if "y" in train_df.columns:
        y = train_df["y"].to_numpy()
        for cols in feature_groups.values():
            for col in cols:
                if col in train_df.columns:
                    x = train_df[col].to_numpy()
                    if x.ndim == 1 and len(x) == len(y):
                        corr = np.corrcoef(x, y)[0, 1]
                        if np.isfinite(corr) and abs(corr) > 0.5:
                            issues.append(
                                LeakageIssue(
                                    level="warning",
                                    message=f"high correlation between {col} and outcome",
                                    locations=[col],
                                )
                            )
    return issues


__all__ = ["check_leakage", "LeakageIssue"]
