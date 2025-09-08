"""Reproducible seeding utilities."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    from hydra import compose, initialize_config_module
except Exception:  # pragma: no cover - hydra not installed
    compose = initialize_config_module = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and environment-level RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:  # pragma: no cover - simple execution
    with initialize_config_module(config_module="engine.config"):
        cfg = compose(config_name="config")
    set_seed(int(cfg.seed))
    print(f"Seed set to {cfg.seed}")


if __name__ == "__main__":  # pragma: no cover
    main()
