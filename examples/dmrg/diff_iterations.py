from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from model_registry import ACTIVE_MODELS

SNAPSHOT_ROOT = Path(__file__).resolve().parent / "snapshots"
SUPPORTED_MODELS = ACTIVE_MODELS


def snapshot_file(model: str, iteration: int) -> Path:
    return SNAPSHOT_ROOT / model / f"iter_{iteration:04d}" / "simple_dmrg.py"


def main():
    parser = argparse.ArgumentParser(description="Show the code diff between two saved DMRG iterations.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--from-iteration", required=True, type=int)
    parser.add_argument("--to-iteration", required=True, type=int)
    args = parser.parse_args()

    left = snapshot_file(args.model, args.from_iteration)
    right = snapshot_file(args.model, args.to_iteration)
    if not left.exists():
        raise SystemExit(f"missing snapshot file: {left}")
    if not right.exists():
        raise SystemExit(f"missing snapshot file: {right}")

    diff = difflib.unified_diff(
        left.read_text().splitlines(keepends=True),
        right.read_text().splitlines(keepends=True),
        fromfile=str(left),
        tofile=str(right),
    )
    print("".join(diff))


if __name__ == "__main__":
    main()
