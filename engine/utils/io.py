"""I/O utilities."""
from __future__ import annotations

from pathlib import Path


def read_text(path: str | Path) -> str:
    """Read a text file."""
    return Path(path).read_text(encoding="utf-8")
