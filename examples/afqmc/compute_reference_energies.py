from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import Any

try:
    from .model_registry import ACTIVE_SYSTEMS
    from .periodic_benchmark import compute_reference_record
except ImportError:
    from model_registry import ACTIVE_SYSTEMS
    from periodic_benchmark import compute_reference_record

OUTPUT_PATH = Path(__file__).resolve().parent / "reference_energies.json"


def _primary_payload(record: dict[str, Any]) -> dict[str, Any]:
    references = record.get("references")
    if isinstance(references, dict) and references:
        primary = record.get("primary_reference")
        if isinstance(primary, str) and isinstance(references.get(primary), dict):
            return dict(references[primary])
    return {
        "reference_method": record.get("reference_method"),
        "reference_energy": record.get("reference_energy"),
        "wall_seconds": record.get("wall_seconds"),
    }


def _merge_reference_record(existing_record: dict[str, Any] | None, new_record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing_record, dict):
        return new_record

    merged = dict(existing_record)
    merged.update({key: value for key, value in new_record.items() if key != "references"})

    existing_sources = existing_record.get("references")
    new_sources = new_record.get("references")
    merged_sources: dict[str, Any] = {}
    if isinstance(existing_sources, dict):
        merged_sources.update(existing_sources)
    if isinstance(new_sources, dict):
        merged_sources.update(new_sources)
    if merged_sources:
        merged["references"] = merged_sources

    preferred_primary = existing_record.get("primary_reference")
    if not (isinstance(preferred_primary, str) and preferred_primary in merged_sources):
        preferred_primary = new_record.get("primary_reference")
    if isinstance(preferred_primary, str):
        merged["primary_reference"] = preferred_primary

    primary = _primary_payload(merged)
    if primary.get("reference_method") is not None:
        merged["reference_method"] = primary["reference_method"]
    if primary.get("reference_energy") is not None:
        merged["reference_energy"] = primary["reference_energy"]
    if primary.get("wall_seconds") is not None:
        merged["wall_seconds"] = primary["wall_seconds"]
    return merged


def main():
    parser = argparse.ArgumentParser(description="Compute unconstrained offline periodic electronic CCSD(T) reference energies.")
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
        existing[system] = _merge_reference_record(existing.get(system), compute_reference_record(system))
    OUTPUT_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps({"reference_file": str(OUTPUT_PATH), "systems": systems}, indent=2))


if __name__ == "__main__":
    main()
