"""
Minimal CUDA-Q VQE benchmark for the fixed LiH external-agent lane.

This LiH file keeps a single explicit CUDA-Q UCCSD-style method active at a
time, but makes it easy to mutate the parameter compression and optimizer
hyperparameters between queued iterations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass

import cudaq
import numpy as np
from cudaq import spin
from openfermion import MolecularData, get_fermion_operator, get_sparse_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from scipy.optimize import minimize

cudaq.set_target("qpp-cpu")

CHEMICAL_ACCURACY = 1e-3


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    geometry: list[tuple[str, tuple[float, float, float]]]
    active_electrons: int
    active_orbitals: int
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"


@dataclass(frozen=True)
class RunConfig:
    name: str
    param_model: str
    optimizer: str
    max_steps: int
    seed: int = 42
    init_scale: float = 0.02
    step_size: float = 0.05
    min_step_size: float = 1e-5
    initial_parameters: tuple[float, ...] = ()
    cobyla_rhobeg: float = 0.1
    cobyla_tol: float = 1e-5
    powell_xtol: float = 1e-4
    powell_ftol: float = 1e-6


MOLECULE_SPECS = {
    "BH": MoleculeSpec(
        name="BH",
        geometry=[("B", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.23))],
        active_electrons=2,
        active_orbitals=3,
    ),
    "LiH": MoleculeSpec(
        name="LiH",
        geometry=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.57))],
        active_electrons=2,
        active_orbitals=4,
    ),
    "BeH2": MoleculeSpec(
        name="BeH2",
        geometry=[("H", (0.0, 0.0, -1.326)), ("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.326))],
        active_electrons=4,
        active_orbitals=4,
    ),
    "H2O": MoleculeSpec(
        name="H2O",
        geometry=[("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.757, 0.587)), ("H", (0.0, -0.757, 0.587))],
        active_electrons=6,
        active_orbitals=4,
    ),
    "N2": MoleculeSpec(
        name="N2",
        geometry=[("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))],
        active_electrons=6,
        active_orbitals=6,
    ),
}
MOLECULE_NAME = "LiH"
SUPPORTED_MOLECULES = (MOLECULE_NAME,)

DEFAULT_CONFIG = RunConfig(
    name="simple_vqe",
    param_model="uccsd_full",
    optimizer="powell",
    max_steps=4096,
    seed=42,
    init_scale=0.0,
    step_size=5e-05,
    min_step_size=1e-08,
    initial_parameters=(
        -0.02630577344889303,
        5.977671899955787e-07,
        -2.6207709899231686e-06,
        -0.02630577344889303,
        5.977671899955787e-07,
        -2.6207709899231686e-06,
        0.03336929846442227,
        1.494177349272587e-06,
        2.6496664189906226e-06,
        -2.349310951693074e-06,
        0.06829227268191392,
        1.8069425535222874e-06,
        -2.853774880094742e-07,
        -1.8082194890298674e-06,
        0.06832947076314003,
    ),
    cobyla_rhobeg=0.005,
    cobyla_tol=1e-08,
    powell_xtol=1e-07,
    powell_ftol=1e-12,
)


@cudaq.kernel
def uccsd_kernel(n_qubits: int, active_electrons: int, theta: list[float]):
    qubits = cudaq.qvector(n_qubits)
    for idx in range(active_electrons):
        x(qubits[idx])
    cudaq.kernels.uccsd(qubits, theta, active_electrons, n_qubits)


class WallTimeReached(RuntimeError):
    pass


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def choose_active_space(molecule: MolecularData, spec: MoleculeSpec) -> tuple[list[int], list[int]]:
    if spec.active_electrons % 2 != 0:
        raise ValueError("active electrons must be even for this restricted-HF benchmark")

    n_core_orbitals = (molecule.n_electrons - spec.active_electrons) // 2
    if n_core_orbitals < 0:
        raise ValueError(f"{spec.name} CAS({spec.active_electrons},{spec.active_orbitals}) freezes a negative number of orbitals")
    if spec.active_electrons > 2 * spec.active_orbitals:
        raise ValueError(f"{spec.name} CAS({spec.active_electrons},{spec.active_orbitals}) overfills the active space")
    if n_core_orbitals + spec.active_orbitals > molecule.n_orbitals:
        raise ValueError(f"{spec.name} CAS({spec.active_electrons},{spec.active_orbitals}) exceeds the STO-3G orbital count")

    occupied_indices = list(range(n_core_orbitals))
    active_indices = list(range(n_core_orbitals, n_core_orbitals + spec.active_orbitals))
    return occupied_indices, active_indices


def openfermion_to_cudaq(qubit_hamiltonian) -> cudaq.SpinOperator:
    operator = 0.0 * spin.i(0)
    for term, coeff in qubit_hamiltonian.terms.items():
        coeff_value = complex(coeff)
        if abs(coeff_value) < 1e-12:
            continue

        if not term:
            operator = operator + float(np.real_if_close(coeff_value)) * spin.i(0)
            continue

        term_operator = None
        for qubit, pauli in term:
            factor = {"X": spin.x(qubit), "Y": spin.y(qubit), "Z": spin.z(qubit)}[pauli]
            term_operator = factor if term_operator is None else term_operator * factor
        operator = operator + float(np.real_if_close(coeff_value)) * term_operator
    return operator


def build_problem(molecule_name: str = MOLECULE_NAME) -> dict:
    if molecule_name != MOLECULE_NAME:
        raise ValueError(f"unsupported molecule {molecule_name!r}; supported values: {SUPPORTED_MOLECULES}")

    spec = MOLECULE_SPECS[MOLECULE_NAME]
    molecule = MolecularData(spec.geometry, spec.basis, spec.multiplicity, spec.charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=False)
    occupied_indices, active_indices = choose_active_space(molecule, spec)

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    qubit_hamiltonian = jordan_wigner(get_fermion_operator(molecular_hamiltonian))
    n_qubits = 2 * spec.active_orbitals
    dense_hamiltonian = np.asarray(get_sparse_operator(qubit_hamiltonian, n_qubits=n_qubits).toarray(), dtype=np.complex128)
    target_energy = float(np.linalg.eigvalsh(dense_hamiltonian).min().real)

    return {
        "name": molecule_name,
        "cas": (spec.active_electrons, spec.active_orbitals),
        "hamiltonian": openfermion_to_cudaq(qubit_hamiltonian),
        "target_energy": target_energy,
        "hf_energy": float(molecule.hf_energy),
        "n_qubits": n_qubits,
        "active_electrons": spec.active_electrons,
        "occupied_indices": occupied_indices,
        "active_indices": active_indices,
        "qubit_hamiltonian_terms": len(qubit_hamiltonian.terms),
    }


def uccsd_counts(problem: dict) -> tuple[int, int]:
    singles = problem["active_electrons"] * (problem["n_qubits"] - problem["active_electrons"]) // 2
    total = int(cudaq.kernels.uccsd_num_parameters(problem["active_electrons"], problem["n_qubits"]))
    return singles, total - singles


def virtual_spatial_orbitals(problem: dict) -> int:
    return (problem["n_qubits"] - problem["active_electrons"]) // 2


def estimate_circuit_depth(_cfg: RunConfig) -> int:
    return 1


def num_params(cfg: RunConfig, problem: dict) -> int:
    singles, doubles = uccsd_counts(problem)
    n_virtual = virtual_spatial_orbitals(problem)
    if cfg.param_model == "uccsd_full":
        return singles + doubles
    if cfg.param_model == "pair_doubles_diag":
        return n_virtual
    if cfg.param_model == "pair_doubles_symmetric":
        return n_virtual + n_virtual * (n_virtual - 1) // 2
    if cfg.param_model == "pair_doubles_full":
        return doubles
    if cfg.param_model == "spin_paired_symmetric":
        return n_virtual + n_virtual + n_virtual * (n_virtual - 1) // 2
    if cfg.param_model == "spin_paired_full":
        return n_virtual + doubles
    raise ValueError(f"unknown parameter model: {cfg.param_model}")


def _symmetric_double_matrix(n_virtual: int, diag: list[float], off_diag: dict[tuple[int, int], float]) -> list[float]:
    values: list[float] = []
    for alpha_index in range(n_virtual):
        for beta_index in range(n_virtual):
            if alpha_index == beta_index:
                values.append(diag[alpha_index])
            else:
                key = tuple(sorted((alpha_index, beta_index)))
                values.append(off_diag[key])
    return values


def expand_params(params: np.ndarray, cfg: RunConfig, problem: dict) -> list[float]:
    raw_params = np.asarray(params, dtype=np.float64)
    singles, doubles = uccsd_counts(problem)
    n_virtual = virtual_spatial_orbitals(problem)

    if cfg.param_model == "uccsd_full":
        return [float(value) for value in raw_params]

    if singles != 2 * n_virtual or doubles != n_virtual * n_virtual:
        raise ValueError(
            f"{cfg.param_model} expects the fixed LiH CAS structure, got singles={singles}, doubles={doubles}, n_virtual={n_virtual}"
        )

    if cfg.param_model == "pair_doubles_diag":
        diag = [float(value) for value in raw_params]
        return [0.0] * singles + _symmetric_double_matrix(n_virtual, diag, {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0})

    if cfg.param_model == "pair_doubles_symmetric":
        diag = [float(value) for value in raw_params[:n_virtual]]
        off_diag = {
            (0, 1): float(raw_params[n_virtual + 0]),
            (0, 2): float(raw_params[n_virtual + 1]),
            (1, 2): float(raw_params[n_virtual + 2]),
        }
        return [0.0] * singles + _symmetric_double_matrix(n_virtual, diag, off_diag)

    if cfg.param_model == "pair_doubles_full":
        return [0.0] * singles + [float(value) for value in raw_params]

    single_angles = [float(value) for value in raw_params[:n_virtual]]
    single_params = single_angles + single_angles
    if cfg.param_model == "spin_paired_symmetric":
        diag_offset = n_virtual
        diag = [float(value) for value in raw_params[diag_offset : diag_offset + n_virtual]]
        off_diag = {
            (0, 1): float(raw_params[diag_offset + n_virtual + 0]),
            (0, 2): float(raw_params[diag_offset + n_virtual + 1]),
            (1, 2): float(raw_params[diag_offset + n_virtual + 2]),
        }
        return single_params + _symmetric_double_matrix(n_virtual, diag, off_diag)
    if cfg.param_model == "spin_paired_full":
        return single_params + [float(value) for value in raw_params[n_virtual:]]
    raise ValueError(f"unknown parameter model: {cfg.param_model}")


def make_initial_parameters(cfg: RunConfig, problem: dict, rng: np.random.Generator) -> np.ndarray:
    n_parameters = num_params(cfg, problem)
    if cfg.initial_parameters:
        x0 = np.asarray(cfg.initial_parameters, dtype=np.float64)
        if x0.size != n_parameters:
            raise ValueError(f"initial_parameters has length {x0.size}, expected {n_parameters}")
        if cfg.init_scale > 0.0:
            x0 = x0 + cfg.init_scale * rng.standard_normal(n_parameters)
        return x0
    if cfg.init_scale == 0.0:
        return np.zeros(n_parameters, dtype=np.float64)
    return cfg.init_scale * rng.standard_normal(n_parameters)


def energy_from_params(params: np.ndarray, cfg: RunConfig, problem: dict) -> float:
    result = cudaq.observe(
        uccsd_kernel,
        problem["hamiltonian"],
        problem["n_qubits"],
        problem["active_electrons"],
        expand_params(params, cfg, problem),
    )
    return float(result.expectation())


def optimizer_options(cfg: RunConfig) -> dict:
    if cfg.optimizer == "cobyla":
        return {
            "maxiter": cfg.max_steps,
            "disp": False,
            "rhobeg": cfg.cobyla_rhobeg,
            "tol": cfg.cobyla_tol,
        }
    if cfg.optimizer == "powell":
        return {
            "maxiter": cfg.max_steps,
            "disp": False,
            "xtol": cfg.powell_xtol,
            "ftol": cfg.powell_ftol,
        }
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def run_coordinate_search(
    cfg: RunConfig,
    current_x: np.ndarray,
    current_energy: float,
    evaluate,
    timed_out,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)
    step_size = cfg.step_size
    passes = 0

    while passes < cfg.max_steps and not timed_out():
        improved = False
        for index in rng.permutation(len(current_x)):
            local_best_energy = current_energy
            local_best_x = None
            for direction in (-1.0, 1.0):
                candidate = current_x.copy()
                candidate[index] += direction * step_size
                value = evaluate(candidate)
                if value < local_best_energy:
                    local_best_energy = value
                    local_best_x = candidate
            if local_best_x is not None:
                current_x = local_best_x
                current_energy = local_best_energy
                improved = True
            if timed_out():
                raise WallTimeReached

        if len(current_x) <= 4 and not timed_out():
            local_best_energy = current_energy
            local_best_x = None
            for directions in itertools.product((-1.0, 1.0), repeat=len(current_x)):
                candidate = current_x + step_size * np.asarray(directions, dtype=np.float64)
                value = evaluate(candidate)
                if value < local_best_energy:
                    local_best_energy = value
                    local_best_x = candidate
                if timed_out():
                    raise WallTimeReached
            if local_best_x is not None:
                current_x = local_best_x
                current_energy = local_best_energy
                improved = True

        passes += 1
        if improved:
            continue
        step_size *= 0.5
        if step_size < cfg.min_step_size:
            break

    return current_x, current_energy


def run_config(
    cfg: RunConfig,
    problem: dict,
    chemical_accuracy: float = CHEMICAL_ACCURACY,
    wall_time_limit: float = 20.0,
) -> dict:
    target_energy = problem["target_energy"]
    rng = np.random.default_rng(cfg.seed)
    x0 = make_initial_parameters(cfg, problem, rng)

    history: list[tuple[int, float, float]] = []
    chem_acc_step = None
    nfev = 0
    start = time.perf_counter()
    best_energy = float("inf")
    best_x = np.zeros_like(x0)
    last_x: np.ndarray | None = None
    last_value: float | None = None

    def timed_out() -> bool:
        return (time.perf_counter() - start) >= wall_time_limit

    def evaluate(x: np.ndarray) -> float:
        nonlocal nfev, best_energy, best_x, chem_acc_step, last_x, last_value
        if timed_out():
            raise WallTimeReached
        trial_x = np.asarray(x, dtype=np.float64)
        if last_x is not None and np.array_equal(trial_x, last_x):
            value = float(last_value)
        else:
            value = energy_from_params(trial_x, cfg, problem)
            last_x = trial_x.copy()
            last_value = value
            nfev += 1
        error = value - target_energy
        history.append((len(history) + 1, value, error))
        if chem_acc_step is None and error <= chemical_accuracy:
            chem_acc_step = len(history)
        if value < best_energy:
            best_energy = value
            best_x = trial_x.copy()
        return float(value)

    reference_x = np.zeros_like(x0)
    current_x = np.array(reference_x, copy=True)
    current_energy = evaluate(reference_x)

    trial_starts: list[np.ndarray] = []
    if cfg.initial_parameters:
        trial_starts.append(np.array(x0, copy=True))
        perturbation = cfg.init_scale * rng.standard_normal(len(x0)) if cfg.init_scale > 0.0 else np.zeros_like(x0)
        if np.any(perturbation):
            trial_starts.append(np.array(x0 + perturbation, copy=True))
            trial_starts.append(np.array(x0 - perturbation, copy=True))
    elif cfg.init_scale > 0.0 and np.any(np.abs(x0) > 0.0):
        trial_starts.append(np.array(x0, copy=True))
        trial_starts.append(np.array(-x0, copy=True))

    for trial_x in trial_starts:
        trial_energy = evaluate(trial_x)
        if trial_energy < current_energy:
            current_x = np.array(trial_x, copy=True)
            current_energy = trial_energy

    result_x = np.array(current_x, copy=True)
    result_fun = float(current_energy)
    try:
        if cfg.optimizer == "coordinate":
            result_x, result_fun = run_coordinate_search(
                cfg=cfg,
                current_x=current_x,
                current_energy=current_energy,
                evaluate=evaluate,
                timed_out=timed_out,
            )
        else:
            result = minimize(
                evaluate,
                x0=current_x,
                method={"cobyla": "COBYLA", "powell": "Powell"}[cfg.optimizer],
                callback=lambda _xk: (_ for _ in ()).throw(WallTimeReached()) if timed_out() else None,
                options=optimizer_options(cfg),
            )
            result_x = np.asarray(result.x, dtype=np.float64)
            result_fun = float(result.fun)
    except WallTimeReached:
        pass

    if result_fun < best_energy:
        best_energy = result_fun
        best_x = result_x

    if not history or abs(history[-1][1] - best_energy) > 1e-10:
        final_error = best_energy - target_energy
        history.append((len(history) + 1, best_energy, final_error))
        if chem_acc_step is None and final_error <= chemical_accuracy:
            chem_acc_step = len(history)

    final_step, final_energy, final_error = history[-1]
    return {
        "config": config_to_dict(cfg),
        "molecule": problem["name"],
        "cas": list(problem["cas"]),
        "target_energy": target_energy,
        "hf_energy": problem["hf_energy"],
        "iterations": final_step,
        "nfev": nfev,
        "chem_acc_step": chem_acc_step,
        "final_energy": final_energy,
        "final_error": final_error,
        "circuit_depth": estimate_circuit_depth(cfg),
        "wall_seconds": time.perf_counter() - start,
        "wall_budget_seconds": wall_time_limit,
        "history": history,
        "occupied_indices": problem["occupied_indices"],
        "active_indices": problem["active_indices"],
        "qubit_hamiltonian_terms": problem["qubit_hamiltonian_terms"],
        "best_parameters": [float(value) for value in best_x],
    }


def compact_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "history"}


def main():
    parser = argparse.ArgumentParser(description="Run the fixed LiH CUDA-Q VQE benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=5.0)
    args = parser.parse_args()

    problem = build_problem(MOLECULE_NAME)
    result = run_config(DEFAULT_CONFIG, problem, chemical_accuracy=CHEMICAL_ACCURACY, wall_time_limit=args.wall_seconds)
    summary = {
        "task": "simple_cudaq_vqe",
        "molecule": MOLECULE_NAME,
        "cas": list(problem["cas"]),
        "metric": "final_energy",
        "lower_is_better": True,
        "score": result["final_energy"],
        "chemical_accuracy": CHEMICAL_ACCURACY,
        "supported_molecules": list(SUPPORTED_MOLECULES),
        **compact_result(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
