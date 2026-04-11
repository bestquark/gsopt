from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .optuna_utils import THREAD_BUDGET_ENV


def run_trial_source(
    *,
    source_file: Path,
    repo_root: Path,
    wall_seconds: float,
    extra_env: dict[str, str] | None = None,
) -> tuple[dict | None, str, str, str | None]:
    env = os.environ.copy()
    env.update(THREAD_BUDGET_ENV)
    if extra_env:
        env.update(extra_env)
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
    except subprocess.TimeoutExpired:
        return None, "", "", f"trial evaluation exceeded the {wall_seconds:.1f}s wall-time budget"
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return None, stdout, stderr, stderr or stdout or "trial evaluation failed"
    if not stdout:
        return None, stdout, stderr, "evaluation produced no stdout"
    try:
        return json.loads(stdout), stdout, stderr, None
    except json.JSONDecodeError as exc:
        return None, stdout, stderr, f"evaluation stdout was not valid JSON: {exc}"
