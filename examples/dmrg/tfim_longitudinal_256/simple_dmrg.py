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
    cutoff: float
    solver_tol: float
    max_sweeps: int
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
MODEL_NAME = "tfim_longitudinal_256"
SUPPORTED_MODELS = (MODEL_NAME,)

DEFAULT_CONFIG = RunConfig(
    name="simple_dmrg",
    bond_schedule=(32, 64, 96, 128, 160, 192, 256),
    cutoff=1e-10,
    solver_tol=1e-6,
    max_sweeps=64,
    init_bond_dim=13,
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
    return (cfg.bond_schedule, cfg.cutoff, cfg.solver_tol, cfg.max_sweeps, cfg.init_bond_dim, cfg.init_seed)


def make_config_name(cfg: RunConfig) -> str:
    schedule = "-".join(map(str, cfg.bond_schedule))
    return (
        f"schedule{schedule}-cut{cfg.cutoff:.0e}-tol{cfg.solver_tol:.0e}-"
        f"sweeps{cfg.max_sweeps}-init{cfg.init_bond_dim}-seed{cfg.init_seed}"
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


def build_initial_state(cfg: RunConfig, problem: dict):
    onsite = -0.5 * np.asarray(qutip.sigmax().full(), dtype=np.float64)
    onsite -= 0.075 * np.asarray(qutip.sigmaz().full(), dtype=np.float64)
    _, eigvecs = np.linalg.eigh(onsite)
    local_state = np.real_if_close(eigvecs[:, 0]).astype(np.float64)
    if local_state[0] < 0.0:
        local_state = -local_state
    state = qtn.MPS_product_state([local_state.copy() for _ in range(problem["chain_length"])], cyclic=problem["cyclic"])
    if cfg.init_bond_dim > 1:
        state.expand_bond_dimension(cfg.init_bond_dim, rand_strength=1e-3, create_bond=True)
    state.left_canonize(normalize=True)
    return state


def run_sweeps(
    solver,
    cfg: RunConfig,
    start: float,
    wall_time_limit: float,
    history: list[tuple[int, float, int]],
    sweep_budget: int,
    bond_cap: int | None = None,
) -> None:
    step_offset = len(history)
    for sweep in range(sweep_budget):
        if (time.perf_counter() - start) >= wall_time_limit:
            break
        step = step_offset + sweep
        direction = "R" if step % 2 == 0 else "L"
        max_bond = cfg.bond_schedule[min(step, len(cfg.bond_schedule) - 1)]
        if bond_cap is not None:
            max_bond = min(max_bond, bond_cap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.solve(
                tol=cfg.solver_tol,
                bond_dims=[max_bond],
                cutoffs=[cfg.cutoff],
                sweep_sequence=direction,
                max_sweeps=1,
                verbosity=0,
                suppress_warnings=True,
            )
        energy = float(np.real(solver.energy))
        history.append((step + 1, energy, int(solver.state.max_bond())))


def run_config(cfg: RunConfig, problem: dict, wall_time_limit: float = 20.0) -> dict:
    mpo = problem["mpo"]
    start = time.perf_counter()
    p0 = build_initial_state(cfg, problem)
    solver2 = qtn.DMRG2(mpo, bond_dims=list(cfg.bond_schedule), cutoffs=cfg.cutoff, p0=p0)
    solver2.opts["local_eig_tol"] = 1e-4
    solver2.opts["local_eig_ncv"] = 8
    history: list[tuple[int, float, int]] = []
    run_sweeps(solver2, cfg, start, wall_time_limit, history, sweep_budget=max(1, cfg.max_sweeps // 2))

    solver = solver2
    if history and (time.perf_counter() - start) < wall_time_limit:
        solver1 = qtn.DMRG1(mpo, bond_dims=[cfg.bond_schedule[-2]], cutoffs=cfg.cutoff, p0=solver2.state)
        solver1.opts["local_eig_tol"] = 5e-5
        solver1.opts["local_eig_ncv"] = 8
        run_sweeps(
            solver1,
            cfg,
            start,
            wall_time_limit,
            history,
            sweep_budget=cfg.max_sweeps,
            bond_cap=cfg.bond_schedule[-2],
        )
        if len(history) > 0:
            solver = solver1

    if not history:
        energy = float(np.real(solver2.energy))
        history.append((1, energy, int(solver2.state.max_bond())))
        solver = solver2

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
