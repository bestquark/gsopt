from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path

from track_iteration import snapshot_source_file

SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_VQE_SNAPSHOT_ROOT", SCRIPT_DIR / "snapshots"))


def molecule_slug(name: str) -> str:
    return name.lower().replace("+", "_plus")


def snapshot_file(molecule: str, iteration: int) -> Path:
    return snapshot_source_file(SNAPSHOT_ROOT / molecule_slug(molecule) / f"iter_{iteration:04d}")


def main():
    parser = argparse.ArgumentParser(description="Show the code diff between two saved VQE iterations.")
    parser.add_argument("--molecule", required=True, choices=["N2", "N2_60", "BH", "LiH", "BeH2", "H2O"])
    parser.add_argument("--from-iteration", required=True, type=int)
    parser.add_argument("--to-iteration", required=True, type=int)
    args = parser.parse_args()

    left = snapshot_file(args.molecule, args.from_iteration)
    right = snapshot_file(args.molecule, args.to_iteration)
    if not left.exists():
        raise SystemExit(f"missing snapshot: {left}")
    if not right.exists():
        raise SystemExit(f"missing snapshot: {right}")

    diff = difflib.unified_diff(
        left.read_text().splitlines(keepends=True),
        right.read_text().splitlines(keepends=True),
        fromfile=str(left),
        tofile=str(right),
    )
    print("".join(diff), end="")


if __name__ == "__main__":
    main()
