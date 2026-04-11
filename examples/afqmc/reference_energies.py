from __future__ import annotations

import json
from pathlib import Path


REFERENCE_FILE = Path(__file__).resolve().parent / "reference_energies.json"


def load_reference_data() -> dict:
    if not REFERENCE_FILE.exists():
        return {}
    text = REFERENCE_FILE.read_text().strip()
    if not text:
        return {}
    return json.loads(text)


def reference_energy(system: str) -> float | None:
    payload = load_reference_data()
    record = payload.get(system)
    if record is None:
        return None
    return float(record["reference_energy"])
