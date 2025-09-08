from __future__ import annotations

"""Lightweight MLflow helper utilities."""

import os
from typing import Dict, Optional

try:  # pragma: no cover - optional import
    import mlflow
except Exception:  # pragma: no cover - fallback
    mlflow = None  # type: ignore


def start_run(
    exp_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
) -> None:
    """Start an MLflow run if MLflow is available."""
    if mlflow is None:  # pragma: no cover - safe guard
        return
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name=run_name, tags=tags)


def log_params(params: Dict[str, object]) -> None:
    """Log parameters to the active run."""
    if mlflow is None or mlflow.active_run() is None:  # pragma: no cover
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the active run."""
    if mlflow is None or mlflow.active_run() is None:  # pragma: no cover
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    """Log an artifact file to the active run."""
    if mlflow is None or mlflow.active_run() is None:  # pragma: no cover
        return
    mlflow.log_artifact(path)


def end_run() -> None:
    """End the active MLflow run."""
    if mlflow is None or mlflow.active_run() is None:  # pragma: no cover
        return
    mlflow.end_run()
