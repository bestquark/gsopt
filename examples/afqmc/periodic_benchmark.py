"""
Minimal periodic electronic benchmark for the AFQMC lane surface.

The path remains `examples/afqmc/`, but the active examples are now periodic
PySCF-PBC electronic systems with a fixed wall-time solver budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pyscf.pbc import gto, mp, scf

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_ROOT = SCRIPT_DIR.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from config_override import load_dataclass_override

try:
    from .model_registry import ACTIVE_SYSTEMS, SYSTEM_SPECS
    from .reference_energies import reference_energy
except ImportError:
    from model_registry import ACTIVE_SYSTEMS, SYSTEM_SPECS
    from reference_energies import reference_energy

CONFIG_OVERRIDE_ENV = "AUTORESEARCH_AFQMC_CONFIG_JSON"


@dataclass(frozen=True)
class RunConfig:
    name: str
    trial: str
    cell_precision: float
    conv_tol: float
    max_cycle: int
    diis_space: int
    level_shift: float
    damping: float
    init_guess: str


REFERENCE_CONFIG = RunConfig(
    name="reference_periodic_scf",
    trial="rhf",
    cell_precision=1e-9,
    conv_tol=1e-10,
    max_cycle=96,
    diis_space=12,
    level_shift=0.0,
    damping=0.0,
    init_guess="minao",
)


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def build_cell(spec, cfg: RunConfig):
    cell = gto.Cell()
    cell.atom = spec.atom
    cell.a = [list(row) for row in spec.lattice_vectors]
    cell.unit = spec.unit
    cell.basis = spec.basis
    cell.pseudo = spec.pseudo
    cell.charge = spec.charge
    cell.spin = spec.spin
    cell.precision = cfg.cell_precision
    cell.verbose = 0
    cell.build()
    return cell


def build_solver(cell, cfg: RunConfig):
    if cfg.trial == "rhf":
        solver = scf.RHF(cell)
    elif cfg.trial == "uhf":
        solver = scf.UHF(cell)
    else:
        raise ValueError(f"unsupported trial {cfg.trial!r}")
    solver.conv_tol = cfg.conv_tol
    solver.max_cycle = cfg.max_cycle
    solver.diis_space = cfg.diis_space
    solver.level_shift = cfg.level_shift
    solver.damp = cfg.damping
    solver.verbose = 0
    return solver


def build_problem(system_name: str) -> dict:
    if system_name not in ACTIVE_SYSTEMS:
        raise ValueError(f"unsupported periodic system {system_name!r}; supported values: {ACTIVE_SYSTEMS}")
    spec = SYSTEM_SPECS[system_name]
    return {
        "name": spec.name,
        "label": spec.label,
        "atom": spec.atom,
        "lattice_vectors": [list(row) for row in spec.lattice_vectors],
        "basis": spec.basis,
        "pseudo": spec.pseudo,
        "charge": spec.charge,
        "spin": spec.spin,
        "unit": spec.unit,
    }


def runtime_config(default_config: RunConfig) -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, default_config, RunConfig)


def run_config(cfg: RunConfig, system_name: str, wall_time_limit: float, target_energy: float) -> dict:
    total_start = time.perf_counter()
    spec = SYSTEM_SPECS[system_name]
    cell = build_cell(spec, cfg)
    build_wall = time.perf_counter() - total_start

    solver = build_solver(cell, cfg)
    dm0 = solver.get_init_guess(key=cfg.init_guess)
    scf_start = time.perf_counter()
    final_energy = float(solver.kernel(dm0=dm0))
    scf_wall = time.perf_counter() - scf_start
    final_error = final_energy - target_energy
    abs_final_error = abs(final_error)
    return {
        "config": config_to_dict(cfg),
        "system": system_name,
        "label": spec.label,
        "basis": spec.basis,
        "pseudo": spec.pseudo,
        "metric": "abs_final_error",
        "lower_is_better": True,
        "score": abs_final_error,
        "reference_energy": target_energy,
        "final_energy": final_energy,
        "final_error": final_error,
        "abs_final_error": abs_final_error,
        "converged": bool(getattr(solver, "converged", False)),
        "scf_cycles": int(getattr(solver, "cycles", 0) or 0),
        "nao": int(cell.nao_nr()),
        "nelectron": int(cell.nelectron),
        "wall_seconds": scf_wall,
        "build_wall_seconds": build_wall,
        "total_wall_seconds": time.perf_counter() - total_start,
        "wall_budget_seconds": wall_time_limit,
        "supported_systems": list(ACTIVE_SYSTEMS),
    }


def compute_reference_record(system_name: str) -> dict:
    spec = SYSTEM_SPECS[system_name]
    start = time.perf_counter()
    cfg = RunConfig(
        name="reference_rhf_mp2",
        trial="rhf",
        cell_precision=REFERENCE_CONFIG.cell_precision,
        conv_tol=REFERENCE_CONFIG.conv_tol,
        max_cycle=REFERENCE_CONFIG.max_cycle,
        diis_space=REFERENCE_CONFIG.diis_space,
        level_shift=REFERENCE_CONFIG.level_shift,
        damping=REFERENCE_CONFIG.damping,
        init_guess=REFERENCE_CONFIG.init_guess,
    )
    cell = build_cell(spec, cfg)
    solver = build_solver(cell, cfg)
    dm0 = solver.get_init_guess(key=cfg.init_guess)
    hf_energy = float(solver.kernel(dm0=dm0))
    mp2_solver = mp.MP2(solver)
    corr_energy, _ = mp2_solver.kernel()
    reference_total = hf_energy + float(corr_energy)
    return {
        "trial": cfg.trial,
        "reference_method": "MP2",
        "scf_energy": hf_energy,
        "correlation_energy": float(corr_energy),
        "reference_energy": reference_total,
        "converged": bool(getattr(solver, "converged", False)),
        "wall_seconds": time.perf_counter() - start,
        "basis": spec.basis,
        "pseudo": spec.pseudo,
        "nao": int(cell.nao_nr()),
        "nelectron": int(cell.nelectron),
        "label": spec.label,
    }


def run_cli(system_name: str, default_config: RunConfig) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed periodic electronic benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    args = parser.parse_args()

    target = reference_energy(system_name)
    if target is None:
        raise SystemExit(f"missing reference energy for {system_name}; run compute_reference_energies.py first")

    result = run_config(runtime_config(default_config), system_name, wall_time_limit=args.wall_seconds, target_energy=target)
    print(json.dumps(result, indent=2))
    return 0
