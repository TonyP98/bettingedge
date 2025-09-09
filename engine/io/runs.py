from __future__ import annotations

from datetime import datetime
from pathlib import Path


def new_run_dir(base: str | Path, div: str) -> Path:
    """Create a new run directory under ``base/div`` with a timestamp."""
    base_path = Path(base)
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_path = base_path / div / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path
