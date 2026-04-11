"""
Minimal DMRG benchmark for external autoresearch agents.

This file keeps the full Hamiltonian definition, DMRG method, and evaluation in
one place. The external-agent workflow uses per-model copies of this file so
multiple agents can edit in parallel in one repo tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import qutip
import quimb.tensor as qtn

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from config_override import load_dataclass_override

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=UserWarning)
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_DMRG_CONFIG_JSON"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    spin: float
    chain_length: int
    bond_terms: tuple[tuple[float, str, str], ...]
    onsite_terms: tuple[tuple[float, str], ...] = ()
    cyclic: bool = False


@dataclass(frozen=True)
class RunConfig:
    name: str
    bond_schedule: tuple[int, ...]
    stage_sweeps: tuple[int, ...]
    cutoff: float
    solver_tol: float
    max_sweeps: int
    sweep_sequence: str
    init_bond_dim: int
    init_seed: int


MODEL_SPECS = {
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
MODEL_NAME = "xxz_gapless_256"
SUPPORTED_MODELS = (MODEL_NAME,)

DEFAULT_CONFIG = RunConfig(
    name="simple_dmrg",
    bond_schedule=(32, 64, 192),
    stage_sweeps=(2, 2, 14),
    cutoff=1e-10,
    solver_tol=1e-6,
    max_sweeps=67,
    sweep_sequence="RL",
    init_bond_dim=32,
    init_seed=42,
)


def spin_ops_qutip(spin: float) -> dict[str, qutip.Qobj]:
    if spin == 0.5:
        sx = 0.5 * qutip.sigmax()
        sy = 0.5 * qutip.sigmay()
        sz = 0.5 * qutip.sigmaz()
    elif spin == 1.0:
        sx = qutip.jmat(1, "x")
        sy = qutip.jmat(1, "y")
        sz = qutip.jmat(1, "z")
    else:
        raise ValueError(f"unsupported spin {spin}")
    return {"sx": sx, "sy": sy, "sz": sz, "sz2": sz @ sz}


def spin_ops_arrays(spin: float) -> dict[str, np.ndarray]:
    return {name: np.asarray(op.full(), dtype=np.complex128) for name, op in spin_ops_qutip(spin).items()}


def config_to_dict(cfg: RunConfig) -> dict:
    payload = asdict(cfg)
    payload["bond_schedule"] = list(cfg.bond_schedule)
    return payload


def config_from_dict(data: dict) -> RunConfig:
    payload = dict(data)
    payload["bond_schedule"] = tuple(payload["bond_schedule"])
    return RunConfig(**payload)


def runtime_config() -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, DEFAULT_CONFIG, RunConfig)


def config_signature(cfg: RunConfig) -> tuple:
    return (
        cfg.bond_schedule,
        cfg.stage_sweeps,
        cfg.cutoff,
        cfg.solver_tol,
        cfg.max_sweeps,
        cfg.sweep_sequence,
        cfg.init_bond_dim,
        cfg.init_seed,
    )


def make_config_name(cfg: RunConfig) -> str:
    schedule = "-".join(map(str, cfg.bond_schedule))
    stage_sweeps = "-".join(map(str, cfg.stage_sweeps))
    return (
        f"schedule{schedule}-cut{cfg.cutoff:.0e}-tol{cfg.solver_tol:.0e}-"
        f"sweeps{cfg.max_sweeps}-stages{stage_sweeps}-seq{cfg.sweep_sequence}-"
        f"init{cfg.init_bond_dim}-seed{cfg.init_seed}"
    )


def build_mpo(spec: ModelSpec):
    arrays = spin_ops_arrays(spec.spin)
    builder = qtn.SpinHam1D(S=spec.spin, cyclic=spec.cyclic)
    for coeff, op_name in spec.onsite_terms:
        builder += (coeff, arrays[op_name])
    for coeff, left_name, right_name in spec.bond_terms:
        builder += (coeff, arrays[left_name], arrays[right_name])
    return builder.build_mpo(spec.chain_length)


def build_problem(model_name: str = MODEL_NAME) -> dict:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unsupported model {model_name!r}; supported values: {SUPPORTED_MODELS}")
    spec = MODEL_SPECS[model_name]
    return {
        "name": spec.name,
        "spin": spec.spin,
        "chain_length": spec.chain_length,
        "cyclic": spec.cyclic,
        "bond_terms": [list(term) for term in spec.bond_terms],
        "onsite_terms": [list(term) for term in spec.onsite_terms],
        "mpo": build_mpo(spec),
    }


def run_config(cfg: RunConfig, problem: dict, wall_time_limit: float = 20.0) -> dict:
    mpo = problem["mpo"]
    start = time.perf_counter()
    p0 = qtn.MPS_rand_state(
        problem["chain_length"],
        bond_dim=cfg.init_bond_dim,
        phys_dim=int(2 * problem["spin"] + 1),
        dtype="float64",
        seed=cfg.init_seed,
        cyclic=problem["cyclic"],
    )
    solver = qtn.DMRG2(mpo, bond_dims=list(cfg.bond_schedule), cutoffs=cfg.cutoff, p0=p0)

    history: list[tuple[int, float, int]] = []
    completed_sweeps = 0
    for max_bond, planned_sweeps in zip(cfg.bond_schedule, cfg.stage_sweeps):
        if (time.perf_counter() - start) >= wall_time_limit:
            break
        sweeps_this_call = min(planned_sweeps, cfg.max_sweeps - completed_sweeps)
        if sweeps_this_call <= 0:
            break
        energy_count_before = len(solver.energies)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.solve(
                tol=cfg.solver_tol,
                bond_dims=[max_bond],
                cutoffs=[cfg.cutoff],
                sweep_sequence=cfg.sweep_sequence,
                max_sweeps=sweeps_this_call,
                verbosity=0,
                suppress_warnings=True,
            )
        new_energies = solver.energies[energy_count_before:]
        if not new_energies:
            break
        realized_bond = int(solver.state.max_bond())
        for offset, energy in enumerate(new_energies, start=1):
            history.append((completed_sweeps + offset, float(np.real(energy)), realized_bond))
        completed_sweeps += len(new_energies)

    if not history:
        energy = float(np.real(solver.energy))
        history.append((1, energy, int(solver.state.max_bond())))

    first_step, first_energy, _ = history[0]
    last_step, final_energy, final_max_bond = history[-1]
    entropy_midchain = None if problem["cyclic"] else float(solver.state.entropy(problem["chain_length"] // 2))
    return {
        "config": config_to_dict(cfg),
        "model": problem["name"],
        "spin": problem["spin"],
        "chain_length": problem["chain_length"],
        "cyclic": problem["cyclic"],
        "iterations": last_step,
        "final_energy": final_energy,
        "energy_per_site": final_energy / problem["chain_length"],
        "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
        "wall_seconds": time.perf_counter() - start,
        "wall_budget_seconds": wall_time_limit,
        "max_bond_realized": final_max_bond,
        "entropy_midchain": entropy_midchain,
        "history": history,
        "bond_terms": problem["bond_terms"],
        "onsite_terms": problem["onsite_terms"],
    }


def compact_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "history"}


def main():
    parser = argparse.ArgumentParser(description="Run the fixed-model DMRG benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=5.0)
    args = parser.parse_args()

    problem = build_problem(MODEL_NAME)
    result = run_config(runtime_config(), problem, wall_time_limit=args.wall_seconds)
    summary = {
        "task": "simple_dmrg_ground_state",
        "model": MODEL_NAME,
        "metric": "final_energy",
        "lower_is_better": True,
        "score": result["final_energy"],
        "supported_models": [MODEL_NAME],
        **compact_result(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
