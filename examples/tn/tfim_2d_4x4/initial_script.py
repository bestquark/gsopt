"""
Minimal tensor-network ground-state benchmark for external autoresearch agents.

This benchmark is broader than the DMRG-only suite: the editable method can
choose between 1D MPS algorithms and small-2D PEPS imaginary-time evolution,
while remaining inside a bounded Quimb-based search space.
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

SCRIPT_DIR = Path(__file__).resolve().parent
TN_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "model_registry.py").exists() else SCRIPT_DIR.parent
if str(TN_ROOT) not in sys.path:
    sys.path.insert(0, str(TN_ROOT))
EXAMPLES_ROOT = TN_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from config_override import load_dataclass_override
from model_registry import MODEL_SPECS

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=UserWarning)
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_TN_CONFIG_JSON"


@dataclass(frozen=True)
class RunConfig:
    name: str
    method: str
    init_state: str
    bond_schedule: tuple[int, ...]
    cutoff: float
    solver_tol: float
    max_sweeps: int
    tau: float
    chi: int
    init_bond_dim: int
    init_seed: int = 42
    local_eig_ncv: int = 4


MODEL_NAME = "tfim_2d_4x4"
SUPPORTED_MODELS = (MODEL_NAME,)

DEFAULT_CONFIG = RunConfig(
    name="simple_tn",
    method="tebd2d",
    init_state="plus",
    bond_schedule=(5,),
    cutoff=5e-7,
    solver_tol=1e-4,
    max_sweeps=24,
    tau=0.1,
    chi=32,
    init_bond_dim=2,
    init_seed=42,
    local_eig_ncv=4,
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


def runtime_config() -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, DEFAULT_CONFIG, RunConfig)


def build_uniform_1d_hamiltonians(spec) -> tuple[qtn.MatrixProductOperator, qtn.LocalHam1D]:
    ops = spin_ops_arrays(spec.spin)
    h2 = np.zeros((ops["sx"].shape[0] ** 2, ops["sx"].shape[0] ** 2), dtype=np.complex128)
    if spec.family in {"heisenberg_xxx", "xxz"}:
        h2 = (
            spec.jxy * np.kron(ops["sx"], ops["sx"])
            + spec.jxy * np.kron(ops["sy"], ops["sy"])
            + spec.jz * np.kron(ops["sz"], ops["sz"])
        )
        h1 = spec.bz * ops["sz"] if abs(spec.bz) > 0.0 else None
    else:
        raise ValueError(f"unsupported 1D family: {spec.family}")

    builder = qtn.SpinHam1D(S=spec.spin, cyclic=spec.cyclic)
    if h1 is not None:
        builder += (spec.bz, ops["sz"])
    if abs(spec.jxy) > 0.0:
        builder += (spec.jxy, ops["sx"], ops["sx"])
        builder += (spec.jxy, ops["sy"], ops["sy"])
    if abs(spec.jz) > 0.0:
        builder += (spec.jz, ops["sz"], ops["sz"])
    mpo = builder.build_mpo(spec.lx)
    ham_local = qtn.LocalHam1D(spec.lx, H2=h2, H1=h1, cyclic=spec.cyclic)
    return mpo, ham_local


def build_2d_local_hamiltonian(spec):
    if spec.family == "tfim":
        return qtn.ham_2d_ising(spec.lx, spec.ly, j=spec.j, bx=spec.bx, cyclic=spec.cyclic)
    if spec.family == "heisenberg_xxx":
        return qtn.ham_2d_heis(spec.lx, spec.ly, j=spec.j, bz=spec.bz, cyclic=spec.cyclic)
    raise ValueError(f"unsupported 2D family: {spec.family}")


def build_problem(model_name: str = MODEL_NAME) -> dict:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unsupported model {model_name!r}; supported values: {SUPPORTED_MODELS}")
    spec = MODEL_SPECS[model_name]
    payload = {
        "name": spec.name,
        "geometry": spec.geometry,
        "family": spec.family,
        "spin": spec.spin,
        "lx": spec.lx,
        "ly": spec.ly,
        "nsites": spec.nsites,
        "cyclic": spec.cyclic,
    }
    if spec.geometry == "1d":
        mpo, ham_local = build_uniform_1d_hamiltonians(spec)
        payload["mpo"] = mpo
        payload["ham_local"] = ham_local
    elif spec.geometry == "2d":
        payload["ham_local"] = build_2d_local_hamiltonian(spec)
    else:
        raise ValueError(f"unsupported geometry: {spec.geometry}")
    return payload


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("zero-norm state vector")
    return vector / norm


def local_state(spec, init_state: str, site_index: int) -> np.ndarray:
    if spec.spin == 0.5:
        up = np.array([1.0, 0.0], dtype=np.float64)
        down = np.array([0.0, 1.0], dtype=np.float64)
        plus = normalized(np.array([1.0, 1.0], dtype=np.float64))
        minus = normalized(np.array([1.0, -1.0], dtype=np.float64))
        if init_state == "product_up":
            return up
        if init_state == "product_down":
            return down
        if init_state == "plus":
            return plus
        if init_state == "minus":
            return minus
        if init_state == "neel":
            return up if site_index % 2 == 0 else down
        if init_state == "checkerboard":
            return up if site_index % 2 == 0 else down
    elif spec.spin == 1.0:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        zero = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        down = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        plus = normalized(np.array([1.0, 1.0, 1.0], dtype=np.float64))
        if init_state == "product_up":
            return up
        if init_state == "product_down":
            return down
        if init_state == "plus":
            return plus
        if init_state == "zero":
            return zero
        if init_state in {"neel", "checkerboard"}:
            return up if site_index % 2 == 0 else down
    raise ValueError(f"unsupported init_state {init_state!r} for spin {spec.spin}")


def build_initial_mps(cfg: RunConfig, problem: dict):
    spec = MODEL_SPECS[problem["name"]]
    phys_dim = int(2 * spec.spin + 1)
    if cfg.init_state == "random":
        return qtn.MPS_rand_state(
            spec.lx,
            bond_dim=cfg.init_bond_dim,
            phys_dim=phys_dim,
            dtype="float64",
            seed=cfg.init_seed,
            cyclic=spec.cyclic,
        )
    arrays = [local_state(spec, cfg.init_state, i).copy() for i in range(spec.lx)]
    state = qtn.MPS_product_state(arrays, cyclic=spec.cyclic)
    if cfg.init_bond_dim > 1:
        state.expand_bond_dimension(cfg.init_bond_dim, rand_strength=0.01, create_bond=True)
    state.left_canonize(normalize=True)
    return state


def build_initial_peps(cfg: RunConfig, problem: dict):
    spec = MODEL_SPECS[problem["name"]]
    phys_dim = int(2 * spec.spin + 1)
    if cfg.init_state == "random":
        return qtn.PEPS.rand(
            spec.lx,
            spec.ly,
            bond_dim=max(1, cfg.init_bond_dim),
            phys_dim=phys_dim,
            dtype="float64",
            seed=cfg.init_seed,
        )
    site_map = {}
    for i in range(spec.lx):
        for j in range(spec.ly):
            flat_index = i * spec.ly + j
            site_map[(i, j)] = local_state(spec, cfg.init_state, flat_index).copy()
    return qtn.PEPS.product_state(site_map, cyclic=spec.cyclic)


def run_dmrg(cfg: RunConfig, problem: dict, wall_time_limit: float) -> dict:
    solver_cls = {"dmrg1": qtn.DMRG1, "dmrg2": qtn.DMRG2}[cfg.method]
    start = time.perf_counter()
    p0 = build_initial_mps(cfg, problem)
    solver = solver_cls(problem["mpo"], bond_dims=[cfg.bond_schedule[0]], cutoffs=cfg.cutoff, p0=p0)
    solver.opts["local_eig_tol"] = cfg.solver_tol
    solver.opts["local_eig_ncv"] = cfg.local_eig_ncv

    history: list[tuple[int, float, int]] = []
    for sweep in range(cfg.max_sweeps):
        if (time.perf_counter() - start) >= wall_time_limit:
            break
        direction = "R" if sweep % 2 == 0 else "L"
        max_bond = cfg.bond_schedule[min(sweep, len(cfg.bond_schedule) - 1)]
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
        history.append((sweep + 1, energy, int(solver.state.max_bond())))

    if not history:
        energy = float(np.real(solver.energy))
        history.append((1, energy, int(solver.state.max_bond())))

    first_step, first_energy, _ = history[0]
    last_step, final_energy, final_max_bond = history[-1]
    return {
        "iterations": last_step,
        "final_energy": final_energy,
        "energy_per_site": final_energy / problem["nsites"],
        "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
        "wall_seconds": time.perf_counter() - start,
        "max_bond_realized": final_max_bond,
        "entropy_midchain": float(solver.state.entropy(problem["lx"] // 2)) if problem["geometry"] == "1d" else None,
        "history": history,
    }


def run_tebd1d(cfg: RunConfig, problem: dict, wall_time_limit: float) -> dict:
    start = time.perf_counter()
    state = build_initial_mps(cfg, problem)
    ham_local = problem["ham_local"]
    history: list[tuple[int, float, int]] = []
    current_time = 0.0
    for step in range(cfg.max_sweeps):
        if (time.perf_counter() - start) >= wall_time_limit:
            break
        max_bond = cfg.bond_schedule[min(step, len(cfg.bond_schedule) - 1)]
        tebd = qtn.TEBD(
            state,
            ham_local,
            dt=cfg.tau,
            split_opts={"cutoff": cfg.cutoff, "max_bond": max_bond},
            imag=True,
            progbar=False,
        )
        tebd.update_to(current_time + cfg.tau, order=2, progbar=False)
        current_time += cfg.tau
        state = tebd.pt
        energy = float(np.real(state.compute_local_expectation(ham_local.terms)))
        history.append((step + 1, energy, int(state.max_bond())))

    if not history:
        energy = float(np.real(state.compute_local_expectation(ham_local.terms)))
        history.append((1, energy, int(state.max_bond())))

    first_step, first_energy, _ = history[0]
    last_step, final_energy, final_max_bond = history[-1]
    return {
        "iterations": last_step,
        "final_energy": final_energy,
        "energy_per_site": final_energy / problem["nsites"],
        "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
        "wall_seconds": time.perf_counter() - start,
        "max_bond_realized": final_max_bond,
        "entropy_midchain": float(state.entropy(problem["lx"] // 2)),
        "history": history,
    }


def run_tebd2d(cfg: RunConfig, problem: dict, wall_time_limit: float) -> dict:
    start = time.perf_counter()
    state = build_initial_peps(cfg, problem)
    ham_local = problem["ham_local"]
    history: list[tuple[int, float, int]] = []
    target_bond = cfg.bond_schedule[-1]
    tau = cfg.tau
    tau_schedule = []
    block_size = max(1, cfg.max_sweeps // 4)
    for step in range(cfg.max_sweeps):
        tau_schedule.append(tau)
        if (step + 1) % block_size == 0:
            tau *= 0.5
    last_step_seconds = None
    tebd = qtn.TEBD2D(
        state,
        ham_local,
        tau=tau_schedule[0],
        D=target_bond,
        cutoff=cfg.cutoff,
        chi=cfg.chi,
        imag=True,
        compute_energy_every=1,
        compute_energy_final=True,
        compute_energy_per_site=False,
        progbar=False,
    )
    for step in range(cfg.max_sweeps):
        elapsed = time.perf_counter() - start
        if elapsed >= wall_time_limit:
            break
        if last_step_seconds is not None and (elapsed + last_step_seconds) >= wall_time_limit:
            break
        step_start = time.perf_counter()
        tebd.evolve(steps=1, tau=tau_schedule[step], progbar=False)
        last_step_seconds = time.perf_counter() - step_start
        history.append((step + 1, float(tebd.energy), target_bond))

    if not history:
        tebd.evolve(steps=1, tau=tau_schedule[0], progbar=False)
        history.append((1, float(tebd.energy), target_bond))

    first_step, first_energy, _ = history[0]
    last_step, _, _ = history[-1]
    _, final_energy, final_max_bond = min(history, key=lambda entry: entry[1])
    return {
        "iterations": last_step,
        "final_energy": final_energy,
        "energy_per_site": final_energy / problem["nsites"],
        "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
        "wall_seconds": time.perf_counter() - start,
        "max_bond_realized": final_max_bond,
        "entropy_midchain": None,
        "history": history,
    }


def run_config(cfg: RunConfig, problem: dict, wall_time_limit: float = 20.0) -> dict:
    geometry = problem["geometry"]
    if cfg.method in {"dmrg1", "dmrg2"}:
        if geometry != "1d":
            raise ValueError(f"method {cfg.method} only supports 1D models")
        payload = run_dmrg(cfg, problem, wall_time_limit)
    elif cfg.method == "tebd1d":
        if geometry != "1d":
            raise ValueError("method tebd1d only supports 1D models")
        payload = run_tebd1d(cfg, problem, wall_time_limit)
    elif cfg.method == "tebd2d":
        if geometry != "2d":
            raise ValueError("method tebd2d only supports 2D models")
        payload = run_tebd2d(cfg, problem, wall_time_limit)
    else:
        raise ValueError(f"unsupported method: {cfg.method}")

    return {
        "config": config_to_dict(cfg),
        "model": problem["name"],
        "geometry": problem["geometry"],
        "spin": problem["spin"],
        "nsites": problem["nsites"],
        "shape": [problem["lx"], problem["ly"]],
        "cyclic": problem["cyclic"],
        "wall_budget_seconds": wall_time_limit,
        **payload,
    }


def compact_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "history"}


def main():
    parser = argparse.ArgumentParser(description="Run the fixed-model tensor-network benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=5.0)
    args = parser.parse_args()

    problem = build_problem(MODEL_NAME)
    result = run_config(runtime_config(), problem, wall_time_limit=args.wall_seconds)
    summary = {
        "task": "simple_tn_ground_state",
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
