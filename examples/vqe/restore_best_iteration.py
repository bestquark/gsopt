from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from track_iteration import molecule_root, resolve_script, snapshot_source_file


def best_kept_iteration(molecule: str) -> int:
    results_file = molecule_root(molecule) / "results.tsv"
    if not results_file.exists():
        raise SystemExit(f"missing results file: {results_file}")
    best_iteration = None
    best_value = None
    with results_file.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            iteration_text = row.get("iteration", "")
            snapshot_dir_text = row.get("snapshot_dir", "")
            if iteration_text == "":
                continue
            try:
                iteration = int(iteration_text)
            except ValueError:
                continue
            value = None
            result_path = Path(snapshot_dir_text) / "result.json"
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text())
                    value = abs(float(payload["final_error"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    value = None
            if value is None:
                metric_text = row.get("abs_final_error", "")
                if metric_text == "":
                    continue
                value = float(metric_text)
            if best_value is None or value < best_value:
                best_iteration = iteration
                best_value = value
    if best_iteration is None:
        raise SystemExit(f"no completed iterations found for {molecule}")
    return best_iteration


def snapshot_file(molecule: str, iteration: int) -> Path:
    return snapshot_source_file(molecule_root(molecule) / f"iter_{iteration:04d}")


def main():
    parser = argparse.ArgumentParser(description="Restore the best archived VQE iteration back into the live script.")
    parser.add_argument("--molecule", required=True, choices=["N2", "N2_60", "BH", "LiH", "BeH2", "H2O"])
    parser.add_argument("--script", required=True, help="Path to the molecule-specific simple_vqe.py file.")
    args = parser.parse_args()

    iteration = best_kept_iteration(args.molecule)
    source = snapshot_file(args.molecule, iteration)
    target = resolve_script(args.script)
    if not source.exists():
        raise SystemExit(f"missing snapshot file: {source}")
    shutil.copy2(source, target)
    print(f"restored iteration {iteration} to {target}")


if __name__ == "__main__":
    main()
