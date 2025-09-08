"""CLI entry-point for fitting the state-space Dixon-Coles model."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from engine.models.dc_state_space import DCStateSpace


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the DC state-space model")
    parser.add_argument("command", choices=["fit"], help="Command to execute")
    parser.add_argument("--data", dest="data_source", help="Path to matches data", default=None)
    args, unknown = parser.parse_known_args()

    if args.command == "fit":
        # In this placeholder implementation we simply create a tiny dataframe.
        if args.data_source:
            try:
                df = pd.read_csv(args.data_source)
            except Exception:  # pragma: no cover - demo robustness
                df = pd.DataFrame(
                    {
                        "home_team": ["A"],
                        "away_team": ["B"],
                        "home_goals": [1],
                        "away_goals": [0],
                    }
                )
        else:
            df = pd.DataFrame(
                {
                    "home_team": ["A"],
                    "away_team": ["B"],
                    "home_goals": [1],
                    "away_goals": [0],
                }
            )
        model = DCStateSpace()
        model.fit(df)
        preds = model.predict_df(df)
        print(json.dumps(preds.to_dict(orient="records"), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
