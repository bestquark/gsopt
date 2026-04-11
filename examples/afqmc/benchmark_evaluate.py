from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.evaluator_utils import resolve_source_file, run_source_script

REFERENCE_FIELDS = {
    "reference_energy",
    "final_error",
    "abs_final_error",
}


def main(*, default_source: str = "initial_script.py") -> int:
    parser = argparse.ArgumentParser(description="Evaluate one periodic electronic benchmark source file.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    args = parser.parse_args()

    source_file = resolve_source_file(Path(__file__).resolve(), default_source)
    try:
        result = run_source_script(source_file, args.wall_seconds)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result["metric"] = "final_energy"
    result["score"] = float(result["final_energy"])
    result.setdefault("lower_is_better", True)
    for key in REFERENCE_FIELDS:
        result.pop(key, None)
    print(json.dumps(result, indent=2))
    return 0
