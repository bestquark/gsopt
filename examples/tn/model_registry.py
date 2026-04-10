from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    geometry: str
    family: str
    spin: float
    lx: int
    ly: int
    cyclic: bool = False
    jxy: float = 0.0
    jz: float = 0.0
    j: float = 0.0
    bx: float = 0.0
    bz: float = 0.0

    @property
    def nsites(self) -> int:
        return self.lx * self.ly


MODEL_SPECS = {
    "heisenberg_xxx_384": ModelSpec(
        name="heisenberg_xxx_384",
        geometry="1d",
        family="heisenberg_xxx",
        spin=0.5,
        lx=384,
        ly=1,
        jxy=1.0,
        jz=1.0,
    ),
    "xxz_gapless_256": ModelSpec(
        name="xxz_gapless_256",
        geometry="1d",
        family="xxz",
        spin=0.5,
        lx=256,
        ly=1,
        jxy=1.0,
        jz=0.9,
    ),
    "spin1_heisenberg_64": ModelSpec(
        name="spin1_heisenberg_64",
        geometry="1d",
        family="heisenberg_xxx",
        spin=1.0,
        lx=64,
        ly=1,
        jxy=1.0,
        jz=1.0,
    ),
    "tfim_2d_4x4": ModelSpec(
        name="tfim_2d_4x4",
        geometry="2d",
        family="tfim",
        spin=0.5,
        lx=4,
        ly=4,
        j=1.0,
        bx=3.05,
    ),
    "heisenberg_2d_4x4": ModelSpec(
        name="heisenberg_2d_4x4",
        geometry="2d",
        family="heisenberg_xxx",
        spin=0.5,
        lx=4,
        ly=4,
        j=1.0,
    ),
}

ACTIVE_MODELS = tuple(MODEL_SPECS)

PRETTY_LABELS = {
    "heisenberg_xxx_384": r"Heisenberg XXX, $L=384$",
    "xxz_gapless_256": r"Gapless XXZ, $L=256$",
    "spin1_heisenberg_64": r"Spin-1 Heisenberg, $L=64$",
    "tfim_2d_4x4": r"2D TFIM, $4{\times}4$",
    "heisenberg_2d_4x4": r"2D Heisenberg, $4{\times}4$",
}
