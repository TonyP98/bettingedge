import os
from pathlib import Path

FORBIDDEN = {"importlib.py", "importlib"}  # file o directory

def test_no_importlib_shadowing():
    root = Path(__file__).resolve().parents[1]  # repo root
    offenders = []
    for name in FORBIDDEN:
        p = root / name
        if p.exists():
            # ignora virtualenv, build, .git, ecc.
            parts = set(p.parts)
            if not (".venv" in parts or "build" in parts or ".git" in parts or "dist" in parts):
                offenders.append(str(p))
    assert not offenders, f"Rimuovi/renomina questi path che oscurano la stdlib: {offenders}"
