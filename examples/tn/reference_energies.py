from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


REFERENCE_PATH = Path(__file__).resolve().parent / "reference_energies.json"


@lru_cache(maxsize=1)
def load_reference_data() -> dict[str, dict]:
    if not REFERENCE_PATH.exists():
        return {}
    return json.loads(REFERENCE_PATH.read_text())


def reference_entry(model: str) -> dict | None:
    return load_reference_data().get(model)


def reference_energy(model: str) -> float | None:
    entry = reference_entry(model)
    if entry is None:
        return None
    return float(entry["reference_energy"])


def reference_mutual_information(model: str) -> list[list[float]] | None:
    entry = reference_entry(model)
    if entry is None:
        return None
    matrix = entry.get("mutual_information_matrix")
    if matrix is None:
        return None
    return [[float(value) for value in row] for row in matrix]
