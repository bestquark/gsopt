from __future__ import annotations

import sys
from pathlib import Path


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


ROOT = _repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchkit.entrypoints import evaluate_main


if __name__ == "__main__":
    raise SystemExit(evaluate_main())
