from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from examples.afqmc.periodic_benchmark import evaluate_source_file
from examples.evaluator_utils import locate_repo_root, resolve_source_file

REFERENCE_FIELDS = {
    "reference_energy",
    "final_error",
    "abs_final_error",
}


def main(*, default_source: str = "initial_script.py") -> int:
    parser = argparse.ArgumentParser(description="Evaluate one periodic electronic benchmark source file.")
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    parser.add_argument("--source-file", help=argparse.SUPPRESS)
    parser.add_argument("--internal-direct", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_file = Path(args.source_file).resolve() if args.source_file else resolve_source_file(Path(__file__).resolve(), default_source)
    if args.internal_direct:
        try:
            result = evaluate_source_file(source_file, args.wall_seconds)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    else:
        repo_root = locate_repo_root(source_file)
        cmd = [
            sys.executable,
            "-m",
            "examples.afqmc.benchmark_evaluate",
            "--internal-direct",
            "--wall-seconds",
            str(args.wall_seconds),
            "--source-file",
            str(source_file),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(float(args.wall_seconds), 0.0) + 1.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"evaluation exceeded the {args.wall_seconds:.1f}s wall-time budget") from exc
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            raise SystemExit(stderr or stdout or "evaluation failed")
        if not stdout:
            raise SystemExit("evaluation produced no stdout")
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"evaluation stdout was not valid JSON: {exc}") from exc

    result["metric"] = "final_energy"
    result["score"] = float(result["final_energy"])
    result.setdefault("lower_is_better", True)
    for key in REFERENCE_FIELDS:
        result.pop(key, None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
