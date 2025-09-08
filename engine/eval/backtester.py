"""Backtesting utilities for betting strategies."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover - placeholder
    """Entry point for the backtester CLI."""
    print("Backtester running with config:")
    print(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
