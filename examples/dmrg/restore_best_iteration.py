from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from model_registry import ACTIVE_MODELS
from track_iteration import model_root, resolve_script

SUPPORTED_MODELS = ACTIVE_MODELS


def best_kept_iteration(model: str) -> int:
    results_file = model_root(model) / "results.tsv"
    if not results_file.exists():
        raise SystemExit(f"missing results file: {results_file}")
    best_iteration = None
    best_value = None
    with results_file.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep" or not row.get("final_energy"):
                continue
            iteration = int(row["iteration"])
            value = float(row["final_energy"])
            if best_value is None or value < best_value:
                best_iteration = iteration
                best_value = value
    if best_iteration is None:
        raise SystemExit(f"no completed iterations found for {model}")
    return best_iteration


def snapshot_file(model: str, iteration: int) -> Path:
    return model_root(model) / f"iter_{iteration:04d}" / "simple_dmrg.py"


def main():
    parser = argparse.ArgumentParser(description="Restore the best archived DMRG iteration into the live script.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--script", required=True, help="Path to the model-specific simple_dmrg.py file.")
    args = parser.parse_args()

    iteration = best_kept_iteration(args.model)
    source = snapshot_file(args.model, iteration)
    target = resolve_script(args.script)
    if not source.exists():
        raise SystemExit(f"missing snapshot file: {source}")
    shutil.copy2(source, target)
    print(f"restored iteration {iteration} to {target}")


if __name__ == "__main__":
    main()
