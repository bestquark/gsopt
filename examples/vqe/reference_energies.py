from __future__ import annotations

REFERENCE_ENERGIES = {
    "BH": -24.775702988648234,
    "LiH": -7.864518501418702,
    "BeH2": -15.566235181521328,
    "H2O": -74.97042716151374,
    "N2": -107.6231017720174,
}

ALIASES = {
    "bh": "BH",
    "lih": "LiH",
    "beh2": "BeH2",
    "h2o": "H2O",
    "n2": "N2",
}


def reference_energy(molecule: str) -> float | None:
    canonical = ALIASES.get(molecule.strip().lower(), molecule.strip())
    value = REFERENCE_ENERGIES.get(canonical)
    if value is None:
        return None
    return float(value)
