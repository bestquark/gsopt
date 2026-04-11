from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _locate_runtime_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skills" / "gsopt" / "SKILL.md").exists():
            return candidate / "skills" / "gsopt"
        run_meta = candidate / "run.json"
        if run_meta.exists():
            try:
                payload = json.loads(run_meta.read_text())
            except json.JSONDecodeError:
                payload = {}
            runtime_root = payload.get("runtime_root")
            if runtime_root:
                return Path(runtime_root).resolve()
    env_runtime = os.environ.get("GSOPT_RUNTIME_ROOT")
    if env_runtime:
        return Path(env_runtime).resolve()
    raise RuntimeError(f"could not locate gsopt runtime root from {start}")

RUNTIME_ROOT = _locate_runtime_root(Path(__file__).resolve())
SCRIPTS_ROOT = RUNTIME_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from gsopt_runtime.entrypoints import evaluate_main


if __name__ == "__main__":
    raise SystemExit(evaluate_main())
