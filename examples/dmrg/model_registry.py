from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    spin: float
    chain_length: int
    bond_terms: tuple[tuple[float, str, str], ...]
    onsite_terms: tuple[tuple[float, str], ...] = ()
    cyclic: bool = False


MODEL_SPECS = {
    "heisenberg_xxx_384": ModelSpec(
        name="heisenberg_xxx_384",
        spin=0.5,
        chain_length=384,
        bond_terms=((1.0, "sx", "sx"), (1.0, "sy", "sy"), (1.0, "sz", "sz")),
    ),
    "xxz_gapless_256": ModelSpec(
        name="xxz_gapless_256",
        spin=0.5,
        chain_length=256,
        bond_terms=((1.0, "sx", "sx"), (1.0, "sy", "sy"), (0.9, "sz", "sz")),
    ),
    "tfim_longitudinal_256": ModelSpec(
        name="tfim_longitudinal_256",
        spin=0.5,
        chain_length=256,
        bond_terms=((-1.0, "sz", "sz"),),
        onsite_terms=((-1.0, "sx"), (-0.15, "sz")),
    ),
    "spin1_heisenberg_64": ModelSpec(
        name="spin1_heisenberg_64",
        spin=1.0,
        chain_length=64,
        bond_terms=((1.0, "sx", "sx"), (1.0, "sy", "sy"), (1.0, "sz", "sz")),
    ),
    "spin1_single_ion_critical_64": ModelSpec(
        name="spin1_single_ion_critical_64",
        spin=1.0,
        chain_length=64,
        bond_terms=((1.0, "sx", "sx"), (1.0, "sy", "sy"), (1.0, "sz", "sz")),
        onsite_terms=((0.968, "sz2"),),
    ),
}

ACTIVE_MODELS = tuple(MODEL_SPECS)

PRETTY_LABELS = {
    "heisenberg_xxx_384": r"Heisenberg XXX, $L=384$",
    "xxz_gapless_256": r"Gapless XXZ, $L=256$",
    "tfim_longitudinal_256": r"Longitudinal TFIM, $L=256$",
    "spin1_heisenberg_64": r"Spin-1 Heisenberg, $L=64$",
    "spin1_single_ion_critical_64": r"Spin-1 single-ion critical, $L=64$",
}
