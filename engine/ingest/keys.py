"""Command-line interface to validate football data key specification."""
from __future__ import annotations

import sys
import argparse
import json
import pathlib
from typing import Any

import yaml
import pandas as pd

from ._keys_schema import FootballDataKeys

# Default path to the YAML specification
DEF_SPEC = pathlib.Path(__file__).resolve().parents[1] / "data" / "specs" / "football_data_keys.yaml"


def load_spec(path: pathlib.Path) -> FootballDataKeys:
    """Load and validate the key specification from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FootballDataKeys(**raw)


def check_sample_csv(spec: FootballDataKeys, csv_path: pathlib.Path) -> dict[str, Any]:
    """Perform a light sanity check of a sample CSV against the spec."""
    df = pd.read_csv(csv_path, nrows=100)
    cols = set(df.columns)
    issues: list[str] = []

    # Example: ensure at least one typical 1X2 pre-match triplet exists
    triplets = (
        {f"{prefix}{suf}" for suf in ("H", "D", "A")}
        for prefix in ("Avg", "Max", "B365", "BW", "IW", "PS")
    )
    has_any_pre = any(t.issubset(cols) for t in triplets)
    if not has_any_pre:
        issues.append(
            "No obvious 1X2 pre-match triplet found in sample (e.g., AvgH/AvgD/AvgA)."
        )

    return {"ok": len(issues) == 0, "issues": issues, "columns": sorted(cols)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        type=pathlib.Path,
        default=DEF_SPEC,
        help="Path to football_data_keys.yaml",
    )
    ap.add_argument(
        "--sample",
        type=pathlib.Path,
        help="Optional sample CSV to sanity-check",
    )
    ap.add_argument(
        "--check", action="store_true", help="Strict validation only"
    )
    args = ap.parse_args(argv)

    rc = 0
    try:
        spec = load_spec(args.spec)
        print("[OK] Loaded spec:", args.spec)
        if args.sample and args.sample.exists():
            res = check_sample_csv(spec, args.sample)
            print(json.dumps({"sample_check": res}, indent=2))
            if not res["ok"]:
                rc = max(rc, 2)  # warning exit
    except Exception as exc:  # pragma: no cover - we handle generically
        print("[ERROR] Spec validation failed:", exc, file=sys.stderr)
        rc = 1

    return rc


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
