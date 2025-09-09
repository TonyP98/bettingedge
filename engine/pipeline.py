from __future__ import annotations

import argparse
from pathlib import Path

from engine.data.loader import unify
from engine.io import persist

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def build_canonical(div: str, seasons: list[str] | None = None) -> Path:
    df = unify(div, seasons)
    out_dir = PROCESSED_DIR / div
    out_path = out_dir / "matches.parquet"
    persist.save_df(df, out_path)
    print(
        f"Saved {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()} -> {out_path}"
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BettingEdge pipeline")
    parser.add_argument("--rebuild-canonical", action="store_true")
    parser.add_argument("--div", required=True)
    parser.add_argument("--seasons", nargs="*", default=[])
    args = parser.parse_args(argv)

    seasons = None if not args.seasons or args.seasons == ["all"] else args.seasons

    if args.rebuild_canonical:
        build_canonical(args.div, seasons)
    else:
        parser.error("No action specified")


if __name__ == "__main__":
    main()
