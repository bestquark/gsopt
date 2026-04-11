from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.evaluator_utils import resolve_source_file, run_source_script
from examples.dmrg.reference_energies import reference_energy


def main(*, default_source: str = "simple_dmrg.py") -> int:
    parser = argparse.ArgumentParser(description="Evaluate one DMRG benchmark source file.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    args = parser.parse_args()

    source_file = resolve_source_file(Path(__file__).resolve(), default_source)
    try:
        result = run_source_script(source_file, args.wall_seconds)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    model = str(result["model"])
    ref_energy = reference_energy(model)
    if ref_energy is not None:
        result["reference_energy"] = ref_energy
        result["excess_energy"] = float(result["final_energy"]) - ref_energy
        result["excess_energy_per_site"] = float(result["energy_per_site"]) - (ref_energy / int(result["chain_length"]))
    result.setdefault("metric", "excess_energy")
    result["score"] = float(result.get("excess_energy", result["final_energy"]))
    result.setdefault("lower_is_better", True)
    print(json.dumps(result, indent=2))
    return 0
