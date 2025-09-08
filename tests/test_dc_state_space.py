import pandas as pd

from engine.models.dc_state_space import DCStateSpace


def test_state_centering():
    df = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_goals": [1, 2],
            "away_goals": [0, 1],
        }
    )
    model = DCStateSpace()
    model.fit(df)
    assert abs(model.atk.mean()) < 1e-8
    assert abs(model.def_.mean()) < 1e-8
    # Prediction should yield probabilities summing to 1
    preds = model.predict_match("A", "B")
    total = preds["ph"] + preds["pd"] + preds["pa"]
    assert abs(total - 1.0) < 1e-6
