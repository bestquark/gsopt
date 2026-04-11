from __future__ import annotations

import os
import sys
from pathlib import Path


def _locate_repo_root(start: Path) -> Path:
    env_benchmark = os.environ.get("GSOPT_BENCHMARK_ROOT")
    candidates = []
    if env_benchmark:
        benchmark_root = Path(env_benchmark).resolve()
        candidates.extend([benchmark_root, *benchmark_root.parents])
    current = start.resolve()
    if current.is_file():
        current = current.parent
    candidates.extend([current, *current.parents])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["GSOPT_BENCHMARK_ROOT"] = str(Path(__file__).resolve().parent)

from examples.tn.benchmark_evaluate import main as benchmark_evaluate_main


if __name__ == "__main__":
    raise SystemExit(benchmark_evaluate_main(default_source="initial_script.py"))
