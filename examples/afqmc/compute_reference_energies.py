from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .model_registry import ACTIVE_SYSTEMS
    from .periodic_benchmark import compute_reference_record
except ImportError:
    from model_registry import ACTIVE_SYSTEMS
    from periodic_benchmark import compute_reference_record

OUTPUT_PATH = Path(__file__).resolve().parent / "reference_energies.json"


def main():
    parser = argparse.ArgumentParser(description="Compute frozen periodic electronic reference energies.")
    parser.add_argument("--system", choices=ACTIVE_SYSTEMS)
    args = parser.parse_args()

    systems = [args.system] if args.system else list(ACTIVE_SYSTEMS)
    existing = {}
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text().strip():
        existing = json.loads(OUTPUT_PATH.read_text())
    # Keep non-AFQMC reference records, but drop retired periodic AFQMC entries.
    existing = {
        key: value
        for key, value in existing.items()
        if not (
            isinstance(value, dict)
            and value.get("basis") == "gth-szv"
            and value.get("pseudo") == "gth-pade"
            and (key.endswith("_pbc") or key == "diamond_prim")
            and key not in ACTIVE_SYSTEMS
        )
    }
    for system in systems:
        print(f"computing reference for {system}...", flush=True)
        existing[system] = compute_reference_record(system)
    OUTPUT_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps({"reference_file": str(OUTPUT_PATH), "systems": systems}, indent=2))


if __name__ == "__main__":
    main()
