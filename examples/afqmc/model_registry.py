from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    atom: str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"


MOLECULE_SPECS = {
    "H2": MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
    ),
    "LiH": MoleculeSpec(
        name="LiH",
        atom="Li 0 0 0; H 0 0 1.57",
    ),
    "BH": MoleculeSpec(
        name="BH",
        atom="B 0 0 0; H 0 0 1.23",
    ),
    "BeH2": MoleculeSpec(
        name="BeH2",
        atom="H 0 0 -1.326; Be 0 0 0; H 0 0 1.326",
    ),
    "HF": MoleculeSpec(
        name="HF",
        atom="H 0 0 0; F 0 0 0.917",
    ),
}

ACTIVE_MOLECULES = tuple(MOLECULE_SPECS)
PRETTY_LABELS = {name: name for name in ACTIVE_MOLECULES}
