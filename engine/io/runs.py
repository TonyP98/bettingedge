from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import persist

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"


def create_run_dir(div: str, base: str | Path | None = None) -> Path:
    """Create a new run directory ``runs/div/run_YYYY-MM-DD_HH-MM-SS``."""

    base_path = Path(base) if base is not None else RUNS_DIR
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_path = base_path / div / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def finalize_run(
    run_dir: Path,
    config: dict,
    metrics: dict,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> None:
    """Persist results and update the division index."""

    persist.save_df(equity_df, run_dir / "equity.csv")
    persist.save_df(trades_df, run_dir / "trades.csv")
    persist.save_json(metrics, run_dir / "metrics.json")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True)

    summary = (
        "# Run Summary\n"
        f"Picks: {len(trades_df)}\n"
        f"CAGR: {metrics.get('cagr', 0.0):.4f}\n"
        f"MaxDD: {metrics.get('maxdd', 0.0):.4f}\n"
        f"Sharpe: {metrics.get('sharpe', 0.0):.4f}\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    div_dir = run_dir.parent
    index_path = div_dir / "index.csv"
    if index_path.exists():
        index_df = pd.read_csv(index_path)
    else:
        index_df = pd.DataFrame(
            columns=[
                "run_id",
                "date",
                "train_until",
                "test_from",
                "picks",
                "cagr",
                "maxdd",
                "sharpe",
            ]
        )

    new_row = {
        "run_id": run_dir.name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "train_until": config.get("train_until"),
        "test_from": config.get("test_from"),
        "picks": len(trades_df),
        "cagr": metrics.get("cagr"),
        "maxdd": metrics.get("maxdd"),
        "sharpe": metrics.get("sharpe"),
    }
    index_df = pd.concat([index_df, pd.DataFrame([new_row])], ignore_index=True)
    persist.save_df(index_df, index_path)


__all__ = ["create_run_dir", "finalize_run"]

