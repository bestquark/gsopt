from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from examples.afqmc.molecular_benchmark import LIVE_OBJECTIVE_METRIC, evaluate_source_file
from examples.evaluator_utils import locate_repo_root, resolve_source_file

REFERENCE_FIELDS = {
    "reference_energy",
    "final_error",
    "abs_final_error",
}

MPI_LAUNCH_ENV = "AUTORESEARCH_AFQMC_MPI_LAUNCH"


def _load_json_from_stdout(stdout: str) -> dict:
    payload = stdout.strip()
    if not payload:
        raise SystemExit("evaluation produced no stdout")
    decoder = json.JSONDecoder()
    for start in range(len(payload)):
        if payload[start] != "{":
            continue
        try:
            result, end = decoder.raw_decode(payload[start:])
        except json.JSONDecodeError:
            continue
        if payload[start + end :].strip():
            continue
        if isinstance(result, dict):
            return result
    raise SystemExit("evaluation stdout was not valid JSON")


def run_source_file(source_file: Path, wall_seconds: float, *, extra_env: dict[str, str] | None = None) -> dict:
    repo_root = locate_repo_root(source_file)
    prefix = shlex.split(os.environ.get(MPI_LAUNCH_ENV, "").strip())
    cmd = [
        *prefix,
        sys.executable,
        "-m",
        "examples.afqmc.benchmark_evaluate",
        "--internal-direct",
        "--wall-seconds",
        str(wall_seconds),
        "--source-file",
        str(source_file),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env={**os.environ, **(extra_env or {})},
            capture_output=True,
            text=True,
            check=False,
            timeout=max(float(wall_seconds), 0.0) + 30.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"evaluation exceeded the {wall_seconds:.1f}s wall-time budget") from exc
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        raise SystemExit(stderr or stdout or "evaluation failed")
    return _load_json_from_stdout(stdout)


def main(*, default_source: str = "initial_script.py") -> int:
    parser = argparse.ArgumentParser(description="Evaluate one molecular AFQMC benchmark source file.")
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    parser.add_argument("--source-file", help=argparse.SUPPRESS)
    parser.add_argument("--internal-direct", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_file = Path(args.source_file).resolve() if args.source_file else resolve_source_file(Path(__file__).resolve(), default_source)
    if args.internal_direct:
        try:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                result = evaluate_source_file(source_file, args.wall_seconds)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if result is None:
            return 0
    else:
        result = run_source_file(source_file, args.wall_seconds)

    result["metric"] = LIVE_OBJECTIVE_METRIC
    result["score"] = float(result["risk_adjusted_energy"])
    result.setdefault("lower_is_better", True)
    for key in REFERENCE_FIELDS:
        result.pop(key, None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
