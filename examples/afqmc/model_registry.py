from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    label: str
    atom: str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"


SYSTEM_SPECS = {
    "h2": MoleculeSpec(
        name="h2",
        label="H2 molecule",
        atom="H 0.0 0.0 0.0; H 0.0 0.0 0.74",
    ),
    "lih": MoleculeSpec(
        name="lih",
        label="LiH molecule",
        atom="Li 0.0 0.0 0.0; H 0.0 0.0 1.57",
    ),
    "h2o": MoleculeSpec(
        name="h2o",
        label="H2O molecule",
        atom="O 0.0 0.0 0.0; H 0.0 0.757 0.587; H 0.0 -0.757 0.587",
    ),
    "n2": MoleculeSpec(
        name="n2",
        label="N2 molecule",
        atom="N 0.0 0.0 0.0; N 0.0 0.0 1.10",
    ),
}

ACTIVE_SYSTEMS = tuple(SYSTEM_SPECS)
PRETTY_LABELS = {name: spec.label for name, spec in SYSTEM_SPECS.items()}
