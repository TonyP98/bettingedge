"""Data ingestion utilities."""
from __future__ import annotations

import argparse
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from .contracts import OddsSchema, ResultsSchema


def load_config():
    """Load Hydra configuration for the engine and data specs."""
    with initialize_config_module(config_module="engine.config"):
        base_cfg = compose(config_name="config")
        data_cfg = compose(config_name="data")
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data = data_cfg
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw data sources.")
    parser.add_argument("--dry-run", action="store_true", help="print inferred schema and column mapping")
    args = parser.parse_args()

    cfg = load_config()

    if args.dry_run:
        print("Results schema:")
        try:
            print(ResultsSchema.to_schema())
        except Exception:
            print(ResultsSchema)
        print("\nOdds schema:")
        try:
            print(OddsSchema.to_schema())
        except Exception:
            print(OddsSchema)
        print("\nColumn mapping:")
        print(OmegaConf.to_yaml(cfg.data.mapping))
    else:  # pragma: no cover - placeholder for real ingestion
        print("Ingestion routine not implemented. Use --dry-run to preview schemas.")


if __name__ == "__main__":  # pragma: no cover
    main()
