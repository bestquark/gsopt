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


@dataclass(frozen=True)
class PeriodicTargetProposal:
    slug: str
    label: str
    family: str
    representative_cell: str
    benchmark_role: str
    notes: str


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

PERIODIC_TARGET_PROPOSALS = (
    PeriodicTargetProposal(
        slug="heg_14e_rs1_gamma",
        label="3D homogeneous electron gas (14 electrons, r_s = 1.0, Gamma twist)",
        family="metallic electron gas",
        representative_cell="Cubic Born-von Karman supercell in a plane-wave basis.",
        benchmark_role="Clean metallic benchmark for trial-state, gauge, and propagation-policy search without electron-ion pseudopotential complexity.",
        notes="Useful when we want a basis-clean periodic AFQMC target whose main difficulty is the metallic projector dynamics.",
    ),
    PeriodicTargetProposal(
        slug="h10_chain_pbc",
        label="Periodic H10 hydrogen chain",
        family="quasi-one-dimensional correlated insulator / crossover system",
        representative_cell="Ten-atom hydrogen chain repeated under Born-von Karman boundary conditions.",
        benchmark_role="Minimal periodic bond-stretching target for testing whether the search discovers better trial states in a strongly correlation-sensitive regime.",
        notes="This keeps the chemistry simple while adding periodicity and near-degeneracy pressure absent in the current small-molecule lane.",
    ),
    PeriodicTargetProposal(
        slug="lih_rocksalt_prim",
        label="LiH rocksalt primitive cell",
        family="ionic band insulator",
        representative_cell="Two-atom primitive cell at a fixed lattice constant, Gamma-point or small twist grid.",
        benchmark_role="Compact ionic solid benchmark where orbital basis, trial family, and stochastic budget all matter but the cell remains small enough for repeated AFQMC scoring.",
        notes="This is a natural first electron-ion periodic target because it is chemically simple and historically common in solid-state AFQMC validation.",
    ),
    PeriodicTargetProposal(
        slug="diamond_prim",
        label="Diamond primitive cell",
        family="covalent semiconductor",
        representative_cell="Two-atom primitive cell at the experimental lattice constant with periodic Gaussian or planewave-derived orbitals.",
        benchmark_role="Small covalent solid benchmark complementary to LiH, with different nodal structure and basis sensitivity.",
        notes="This gives the periodic lane a canonical semiconductor target rather than only metallic or ionic cases.",
    ),
)
