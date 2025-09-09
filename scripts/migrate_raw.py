from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    raw_dir = Path("data/raw")
    target = raw_dir / "I1"
    target.mkdir(parents=True, exist_ok=True)

    moved = []
    for csv in raw_dir.glob("*.csv"):
        dest = target / csv.name
        shutil.move(str(csv), dest)
        moved.append(csv.name)

    if moved:
        print("Moved:")
        for name in moved:
            print(f" - {name}")
    else:
        print("No files to migrate")


if __name__ == "__main__":
    main()
