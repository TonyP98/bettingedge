"""Simple rule extraction utilities for the edge playbook."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

__all__ = ["Rule", "fit_rule_playbook"]


@dataclass
class Rule:
    """Container for a single decision rule."""

    rule_id: str
    rule: str
    depth: int
    support: int
    coverage: float
    precision: float
    lift: float
    fold: int | None = None


def _walk_tree(tree, feature_names: List[str]):
    """Yield rule strings and depth from a fitted ``DecisionTreeClassifier``."""

    def recurse(node: int, conds: list[str]):
        if tree.tree_.feature[node] == -2:  # leaf
            rule = " & ".join(conds) if conds else "True"
            yield rule, len(conds)
        else:
            feat = feature_names[tree.tree_.feature[node]]
            thr = tree.tree_.threshold[node]
            left_conds = conds + [f"{feat} <= {thr:.3f}"]
            right_conds = conds + [f"{feat} > {thr:.3f}"]
            yield from recurse(tree.tree_.children_left[node], left_conds)
            yield from recurse(tree.tree_.children_right[node], right_conds)

    yield from recurse(0, [])


def fit_rule_playbook(
    X: np.ndarray,
    y_edge: np.ndarray,
    feature_names: List[str],
    *,
    min_coverage: float = 0.02,
    max_depth: int = 4,
    n_rules: int = 100,
) -> List[Rule]:
    """Fit a simple decision tree and extract rules meeting the constraints."""
    n_samples = X.shape[0]
    # fit a flexible tree and enforce coverage constraints post-hoc
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=1,
        random_state=0,
        class_weight="balanced",
        criterion="entropy",
    )
    clf.fit(X, y_edge)
    base_rate = y_edge.mean() if len(y_edge) else 0.0
    dfX = pd.DataFrame(X, columns=feature_names)
    rules: List[Rule] = []
    for idx, (rule_str, depth) in enumerate(_walk_tree(clf, feature_names)):
        mask = np.ones(len(dfX), dtype=bool)
        if rule_str != "True":
            for cond in rule_str.split(" & "):
                feat, op, thr = cond.split(" ")
                thr = float(thr)
                if op == "<=":
                    mask &= dfX[feat] <= thr
                else:
                    mask &= dfX[feat] > thr
        support = int(mask.sum())
        coverage = support / n_samples
        if coverage < min_coverage or depth == 0:
            continue
        precision = float(y_edge[mask].mean()) if support else 0.0
        lift = precision / base_rate if base_rate else np.nan
        rules.append(
            Rule(
                rule_id=str(idx),
                rule=rule_str,
                depth=depth,
                support=support,
                coverage=coverage,
                precision=precision,
                lift=lift,
            )
        )
    rules.sort(key=lambda r: r.precision * r.coverage, reverse=True)
    return rules[:n_rules]
