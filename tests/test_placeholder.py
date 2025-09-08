"""Placeholder tests for CI."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from engine.utils.seed import set_seed


def test_set_seed_runs() -> None:
    set_seed(0)
