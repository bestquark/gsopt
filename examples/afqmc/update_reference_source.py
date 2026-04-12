from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .compute_reference_energies import OUTPUT_PATH, _merge_reference_record
    from .model_registry import ACTIVE_SYSTEMS
except ImportError:
    from compute_reference_energies import OUTPUT_PATH, _merge_reference_record
    from model_registry import ACTIVE_SYSTEMS


def main():
    parser = argparse.ArgumentParser(description="Merge a named offline AFQMC reference source into reference_energies.json.")
    parser.add_argument("--system", required=True, choices=ACTIVE_SYSTEMS)
    parser.add_argument("--method-key", required=True, help="Stable machine key, for example ph_afqmc or ccsd_t.")
    parser.add_argument("--method-label", required=True, help="Display label, for example ph-AFQMC.")
    parser.add_argument("--energy", required=True, type=float, help="Reference total energy in Hartree.")
    parser.add_argument("--stderr", type=float, help="Optional statistical error bar in Hartree.")
    parser.add_argument("--wall-seconds", type=float, help="Optional total wall time for the reference calculation.")
    parser.add_argument("--primary", action="store_true", help="Promote this source to the primary offline reference.")
    args = parser.parse_args()

    existing = {}
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text().strip():
        existing = json.loads(OUTPUT_PATH.read_text())

    method_key = args.method_key.strip()
    new_record = {
        "primary_reference": method_key if args.primary else existing.get(args.system, {}).get("primary_reference"),
        "references": {
            method_key: {
                "method_key": method_key,
                "reference_method": args.method_label,
                "reference_energy": args.energy,
                **({"stderr": args.stderr} if args.stderr is not None else {}),
                **({"wall_seconds": args.wall_seconds} if args.wall_seconds is not None else {}),
            }
        },
    }
    merged = _merge_reference_record(existing.get(args.system), new_record)
    existing[args.system] = merged
    OUTPUT_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps({"reference_file": str(OUTPUT_PATH), "system": args.system, "method_key": method_key}, indent=2))


if __name__ == "__main__":
    main()
