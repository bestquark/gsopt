from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

THREAD_BUDGET_ENV = {
    "OMP_NUM_THREADS": "10",
    "OPENBLAS_NUM_THREADS": "10",
    "MKL_NUM_THREADS": "10",
    "VECLIB_MAXIMUM_THREADS": "10",
    "NUMEXPR_NUM_THREADS": "10",
    "BLIS_NUM_THREADS": "10",
}


def locate_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    env_benchmark = os.environ.get("GSOPT_BENCHMARK_ROOT")
    if env_benchmark:
        benchmark_root = Path(env_benchmark).resolve()
        for candidate in (benchmark_root, *benchmark_root.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
                return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


def resolve_source_file(current_file: Path, default_name: str) -> Path:
    env_source = os.environ.get("GSOPT_SOURCE_FILE")
    if env_source:
        return Path(env_source).resolve()
    env_benchmark = os.environ.get("GSOPT_BENCHMARK_ROOT")
    if env_benchmark:
        return (Path(env_benchmark).resolve() / default_name).resolve()
    return current_file.with_name(default_name).resolve()


def run_source_script(source_file: Path, wall_seconds: float) -> dict:
    repo_root = locate_repo_root(source_file)
    env = os.environ.copy()
    env.update(THREAD_BUDGET_ENV)
    try:
        proc = subprocess.run(
            [sys.executable, str(source_file), "--wall-seconds", str(wall_seconds)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(float(wall_seconds), 0.0) + 1.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"evaluation exceeded the {wall_seconds:.1f}s wall-time budget") from exc
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        raise RuntimeError(stderr or stdout or "evaluation failed")
    if not stdout:
        raise RuntimeError("evaluation produced no stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"evaluation stdout was not valid JSON: {exc}") from exc
