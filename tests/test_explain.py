import numpy as np
import pandas as pd
import pkgutil, pytest

HAS_SHAP = pkgutil.find_loader("shap") is not None
HAS_LGBM = pkgutil.find_loader("lightgbm") is not None

pytestmark = [
    pytest.mark.skipif(not HAS_SHAP, reason="shap non installato"),
    pytest.mark.skipif(not HAS_LGBM, reason="lightgbm non installato"),
]

if HAS_SHAP:
    import shap
if HAS_LGBM:
    import lightgbm as lgbm

imodels = pytest.importorskip(
    "imodels",
    reason="imodels opzionale; se manca si skippa",
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from engine.explain.shap_utils import shap_global, shap_local
from engine.explain.rulefit import fit_rule_playbook
from engine.explain.whatif import simulate_delta


def test_shap_global_local():
    X, y = make_classification(
        n_samples=200, n_features=3, n_informative=1, random_state=0, n_clusters_per_class=1
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_bg, X_eval, y_bg, y_eval = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    model = lgbm.LGBMClassifier(n_estimators=50, random_state=0)
    model.fit(X_bg, y_bg)

    df_global = shap_global(model, pd.DataFrame(X_bg, columns=feature_names), pd.DataFrame(X_eval, columns=feature_names), feature_names, fold=0)
    top_feature = df_global.sort_values("mean_abs", ascending=False).iloc[0]["feature"]
    assert top_feature == "f0"

    match_ids = [f"m{i}" for i in range(len(X_eval))]
    df_local = shap_local(
        model,
        pd.DataFrame(X_bg, columns=feature_names),
        pd.DataFrame(X_eval[:1], columns=feature_names),
        match_ids[:1],
        feature_names,
        fold=0,
        topk=3,
    )
    explainer = shap.TreeExplainer(
        model, pd.DataFrame(X_bg, columns=feature_names), model_output="probability"
    )
    shap_vals = explainer.shap_values(
        pd.DataFrame(X_eval[:1], columns=feature_names)
    )
    base = explainer.expected_value
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
        base = base[1]
    pred = model.predict_proba(X_eval[:1])[0, 1]
    assert np.isclose(df_local["shap_value"].sum() + base, pred, atol=1e-2)


def test_rulefit_recovers_rule():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["x1", "x2", "x3"])
    y = ((X["x1"] > 0.5) & (X["x2"] < 0)).astype(int)
    rules = fit_rule_playbook(X.values, y.values, list(X.columns), min_coverage=0.05)
    found = False
    for r in rules:
        if "x1 >" in r.rule and "x2 <=" in r.rule and r.precision > 0.7 and r.coverage > 0.1:
            found = True
            break
    assert found


def test_whatif_delta_sign():
    X = pd.DataFrame({"x1": [0.0, 1.0], "x2": [1.0, 1.0]})
    y = np.array([0, 1])
    model = LogisticRegression().fit(X, y)
    row = X.iloc[0].copy()
    row["odds"] = 2.0
    result = simulate_delta(row, "x1", 1.0, model, ["x1", "x2"])
    assert result["delta_p"] >= 0
    assert result["delta_edge"] >= 0
