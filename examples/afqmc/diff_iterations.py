from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path

from track_iteration import snapshot_source_file


def main():
    parser = argparse.ArgumentParser(description="Diff two archived AFQMC iterations.")
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    args = parser.parse_args()

    snapshot_root = Path(
        os.environ.get("AUTORESEARCH_AFQMC_SNAPSHOT_ROOT", Path(__file__).resolve().parent / "snapshots")
    )
    root = snapshot_root / args.molecule.lower().replace("+", "_plus")
    left = snapshot_source_file(root / f"iter_{args.left:04d}")
    right = snapshot_source_file(root / f"iter_{args.right:04d}")
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
    print("".join(diff), end="")


if __name__ == "__main__":
    main()
