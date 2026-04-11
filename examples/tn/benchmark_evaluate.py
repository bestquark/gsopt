from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.evaluator_utils import resolve_source_file, run_source_script


def main(*, default_source: str = "initial_script.py") -> int:
    parser = argparse.ArgumentParser(description="Evaluate one TN benchmark source file.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    args = parser.parse_args()

    source_file = resolve_source_file(Path(__file__).resolve(), default_source)
    result = run_source_script(source_file, args.wall_seconds)
    result.setdefault("metric", "final_energy")
    result["score"] = float(result["final_energy"])
    result.setdefault("lower_is_better", True)
    print(json.dumps(result, indent=2))
    return 0
