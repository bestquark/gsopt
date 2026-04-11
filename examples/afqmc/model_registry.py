from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodicSystemSpec:
    name: str
    label: str
    atom: str
    lattice_vectors: tuple[tuple[float, float, float], ...]
    basis: str = "gth-szv"
    pseudo: str = "gth-pade"
    charge: int = 0
    spin: int = 0
    unit: str = "B"


SYSTEM_SPECS = {
    "h8_cube_pbc": PeriodicSystemSpec(
        name="h8_cube_pbc",
        label="H8 cube supercell",
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 1.4 0; H 0 1.4 1.4; "
            "H 1.4 0 0; H 1.4 0 1.4; H 1.4 1.4 0; H 1.4 1.4 1.4"
        ),
        lattice_vectors=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)),
    ),
    "h10_chain_pbc": PeriodicSystemSpec(
        name="h10_chain_pbc",
        label="H10 chain supercell",
        atom="; ".join(f"H 0 0 {index * 1.4}" for index in range(10)),
        lattice_vectors=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 28.0)),
    ),
    "lih_cubic_pbc": PeriodicSystemSpec(
        name="lih_cubic_pbc",
        label="LiH cubic supercell",
        atom="Li 0 0 0; H 0 0 3.0",
        lattice_vectors=((7.8, 0.0, 0.0), (0.0, 7.8, 0.0), (0.0, 0.0, 7.8)),
    ),
    "diamond_prim": PeriodicSystemSpec(
        name="diamond_prim",
        label="Diamond primitive cell",
        atom="C 0 0 0; C 0.8925 0.8925 0.8925",
        lattice_vectors=((0.0, 1.785, 1.785), (1.785, 0.0, 1.785), (1.785, 1.785, 0.0)),
    ),
}

ACTIVE_SYSTEMS = tuple(SYSTEM_SPECS)
PRETTY_LABELS = {name: spec.label for name, spec in SYSTEM_SPECS.items()}
