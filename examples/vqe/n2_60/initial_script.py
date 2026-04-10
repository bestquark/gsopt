"""
Auxiliary CUDA-Q VQE benchmark for the fixed N2_60 external-agent lane.

This file keeps one explicit CUDA-Q UCCSD-style method active at a time while
making it easy for queued outer-loop mutations to change the compressed
parameter model, optimizer, and warm start.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import cudaq
import numpy as np
from cudaq import spin
from cudaq.kernels.uccsd import uccsd_get_excitation_list
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
    step_size: float = 0.12
    min_step_size: float = 1e-4
    initial_parameters: tuple[float, ...] = ()
    cobyla_rhobeg: float = 0.08
    cobyla_tol: float = 1e-6
    powell_xtol: float = 5e-4
    powell_ftol: float = 1e-7


MOLECULE_SPEC = MoleculeSpec(
    name="N2",
    geometry=[("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))],
    active_electrons=6,
    active_orbitals=6,
)
MOLECULE_NAME = MOLECULE_SPEC.name
SUPPORTED_MOLECULES = (MOLECULE_NAME,)

DEFAULT_CONFIG = RunConfig(
    name='simple_vqe',
    param_model='pair_matrix_exchange_sd',
    optimizer='coordinate',
    max_steps=640,
    seed=4043,
    init_scale=0.003,
    step_size=0.04,
    min_step_size=1e-05,
    initial_parameters=(),
    cobyla_rhobeg=0.04,
    cobyla_tol=1e-07,
    powell_xtol=0.0002,
    powell_ftol=1e-08,
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


def build_uccsd_structure(active_electrons: int, n_qubits: int) -> dict:
    n_occupied = active_electrons // 2
    n_virtual = n_qubits // 2 - n_occupied
    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = uccsd_get_excitation_list(
        active_electrons, n_qubits
    )

    occupied_alpha_indices = [2 * idx for idx in range(n_occupied)]
    occupied_beta_indices = [2 * idx + 1 for idx in range(n_occupied)]
    virtual_alpha_indices = [active_electrons + 2 * idx for idx in range(n_virtual)]
    virtual_beta_indices = [active_electrons + 2 * idx + 1 for idx in range(n_virtual)]

    occupied_alpha_lookup = {orbital: idx for idx, orbital in enumerate(occupied_alpha_indices)}
    occupied_beta_lookup = {orbital: idx for idx, orbital in enumerate(occupied_beta_indices)}
    virtual_alpha_lookup = {orbital: idx for idx, orbital in enumerate(virtual_alpha_indices)}
    virtual_beta_lookup = {orbital: idx for idx, orbital in enumerate(virtual_beta_indices)}

    occupied_pairs = [(left, right) for left in range(n_occupied) for right in range(left + 1, n_occupied)]
    virtual_pairs = [(left, right) for left in range(n_virtual) for right in range(left + 1, n_virtual)]
    occupied_pair_lookup = {pair: idx for idx, pair in enumerate(occupied_pairs)}
    virtual_pair_lookup = {pair: idx for idx, pair in enumerate(virtual_pairs)}

    return {
        "n_occupied": n_occupied,
        "n_virtual": n_virtual,
        "pair_term_count": n_occupied * n_virtual,
        "exchange_term_count": len(occupied_pairs) * len(virtual_pairs),
        "full_parameter_count": int(cudaq.kernels.uccsd_num_parameters(active_electrons, n_qubits)),
        "singles_alpha_spatial": [
            (occupied_alpha_lookup[p_occ], virtual_alpha_lookup[q_virt]) for p_occ, q_virt in singles_alpha
        ],
        "singles_beta_spatial": [
            (occupied_beta_lookup[p_occ], virtual_beta_lookup[q_virt]) for p_occ, q_virt in singles_beta
        ],
        "mixed_double_spatial": [
            (
                occupied_alpha_lookup[p_occ],
                occupied_beta_lookup[q_occ],
                virtual_alpha_lookup[s_virt],
                virtual_beta_lookup[r_virt],
            )
            for p_occ, q_occ, r_virt, s_virt in doubles_mixed
        ],
        "alpha_double_spatial": [
            (
                occupied_alpha_lookup[p_occ],
                occupied_alpha_lookup[q_occ],
                virtual_alpha_lookup[r_virt],
                virtual_alpha_lookup[s_virt],
            )
            for p_occ, q_occ, r_virt, s_virt in doubles_alpha
        ],
        "beta_double_spatial": [
            (
                occupied_beta_lookup[p_occ],
                occupied_beta_lookup[q_occ],
                virtual_beta_lookup[r_virt],
                virtual_beta_lookup[s_virt],
            )
            for p_occ, q_occ, r_virt, s_virt in doubles_beta
        ],
        "occupied_pair_lookup": occupied_pair_lookup,
        "virtual_pair_lookup": virtual_pair_lookup,
    }


def build_problem(molecule_name: str = MOLECULE_NAME) -> dict:
    if molecule_name != MOLECULE_NAME:
        raise ValueError(f"unsupported molecule {molecule_name!r}; supported values: {SUPPORTED_MOLECULES}")

    spec = MOLECULE_SPEC
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
        "uccsd_structure": build_uccsd_structure(spec.active_electrons, n_qubits),
    }


def estimate_circuit_depth(_cfg: RunConfig) -> int:
    return 1


def num_params(cfg: RunConfig, problem: dict) -> int:
    structure = problem["uccsd_structure"]
    pair_term_count = structure["pair_term_count"]
    if cfg.param_model == "pair_matrix_doubles":
        return pair_term_count
    if cfg.param_model == "pair_matrix_sd":
        return 2 * pair_term_count
    if cfg.param_model == "pair_matrix_exchange_sd":
        return 2 * pair_term_count + structure["exchange_term_count"]
    if cfg.param_model == "uccsd_full":
        return structure["full_parameter_count"]
    raise ValueError(f"unknown parameter model: {cfg.param_model}")


def default_initial_parameters(cfg: RunConfig, problem: dict) -> np.ndarray:
    n_parameters = num_params(cfg, problem)
    values = np.zeros(n_parameters, dtype=np.float64)
    if cfg.param_model == "uccsd_full":
        return values

    structure = problem["uccsd_structure"]
    n_virtual = structure["n_virtual"]
    pair_offset = 0 if cfg.param_model == "pair_matrix_doubles" else structure["pair_term_count"]
    for spatial_idx in range(min(structure["n_occupied"], n_virtual)):
        values[pair_offset + spatial_idx * n_virtual + spatial_idx] = -0.05
    return values


def make_initial_parameters(cfg: RunConfig, problem: dict, rng: np.random.Generator) -> np.ndarray:
    n_parameters = num_params(cfg, problem)
    if cfg.initial_parameters:
        x0 = np.asarray(cfg.initial_parameters, dtype=np.float64)
        if x0.size != n_parameters:
            raise ValueError(
                f"initial_parameters has length {x0.size}, expected {n_parameters} for parameter model {cfg.param_model}"
            )
    else:
        x0 = default_initial_parameters(cfg, problem)

    if cfg.init_scale > 0.0:
        x0 = x0 + cfg.init_scale * rng.standard_normal(n_parameters)
    return x0


def reshape_pair_matrix(flat_values: np.ndarray, structure: dict) -> np.ndarray:
    return np.asarray(flat_values, dtype=np.float64).reshape((structure["n_occupied"], structure["n_virtual"]))


def reshape_exchange_matrix(flat_values: np.ndarray, structure: dict) -> np.ndarray:
    return np.asarray(flat_values, dtype=np.float64).reshape(
        (len(structure["occupied_pair_lookup"]), len(structure["virtual_pair_lookup"]))
    )


def mixed_double_amplitude(
    occupied_i: int,
    occupied_j: int,
    virtual_a: int,
    virtual_b: int,
    pair_matrix: np.ndarray,
    exchange_matrix: np.ndarray | None,
    structure: dict,
) -> float:
    base = 0.25 * (
        pair_matrix[occupied_i, virtual_a]
        + pair_matrix[occupied_i, virtual_b]
        + pair_matrix[occupied_j, virtual_a]
        + pair_matrix[occupied_j, virtual_b]
    )
    if exchange_matrix is None or occupied_i == occupied_j or virtual_a == virtual_b:
        return float(base)

    occupied_pair = tuple(sorted((occupied_i, occupied_j)))
    virtual_pair = tuple(sorted((virtual_a, virtual_b)))
    correction = exchange_matrix[
        structure["occupied_pair_lookup"][occupied_pair],
        structure["virtual_pair_lookup"][virtual_pair],
    ]
    return float(base + correction)


def same_spin_double_amplitude(
    occupied_i: int,
    occupied_j: int,
    virtual_a: int,
    virtual_b: int,
    pair_matrix: np.ndarray,
    exchange_matrix: np.ndarray | None,
    structure: dict,
) -> float:
    value = 0.5 * (
        pair_matrix[occupied_i, virtual_a]
        + pair_matrix[occupied_j, virtual_b]
        - pair_matrix[occupied_i, virtual_b]
        - pair_matrix[occupied_j, virtual_a]
    )
    if exchange_matrix is not None:
        value += exchange_matrix[
            structure["occupied_pair_lookup"][(occupied_i, occupied_j)],
            structure["virtual_pair_lookup"][(virtual_a, virtual_b)],
        ]
    return float(value)


def expand_params(params: np.ndarray, cfg: RunConfig, problem: dict) -> list[float]:
    raw_params = np.asarray(params, dtype=np.float64)
    structure = problem["uccsd_structure"]
    if cfg.param_model == "uccsd_full":
        return [float(value) for value in raw_params]

    pair_term_count = structure["pair_term_count"]
    cursor = 0
    if cfg.param_model == "pair_matrix_doubles":
        singles_matrix = np.zeros((structure["n_occupied"], structure["n_virtual"]), dtype=np.float64)
    else:
        singles_matrix = reshape_pair_matrix(raw_params[cursor : cursor + pair_term_count], structure)
        cursor += pair_term_count

    pair_matrix = reshape_pair_matrix(raw_params[cursor : cursor + pair_term_count], structure)
    cursor += pair_term_count

    exchange_matrix = None
    if cfg.param_model == "pair_matrix_exchange_sd":
        exchange_matrix = reshape_exchange_matrix(raw_params[cursor:], structure)

    expanded: list[float] = []
    for occupied_i, virtual_a in structure["singles_alpha_spatial"]:
        expanded.append(float(singles_matrix[occupied_i, virtual_a]))
    for occupied_i, virtual_a in structure["singles_beta_spatial"]:
        expanded.append(float(singles_matrix[occupied_i, virtual_a]))
    for occupied_i, occupied_j, virtual_a, virtual_b in structure["mixed_double_spatial"]:
        expanded.append(
            mixed_double_amplitude(
                occupied_i,
                occupied_j,
                virtual_a,
                virtual_b,
                pair_matrix,
                exchange_matrix,
                structure,
            )
        )
    for occupied_i, occupied_j, virtual_a, virtual_b in structure["alpha_double_spatial"]:
        expanded.append(
            same_spin_double_amplitude(
                occupied_i,
                occupied_j,
                virtual_a,
                virtual_b,
                pair_matrix,
                exchange_matrix,
                structure,
            )
        )
    for occupied_i, occupied_j, virtual_a, virtual_b in structure["beta_double_spatial"]:
        expanded.append(
            same_spin_double_amplitude(
                occupied_i,
                occupied_j,
                virtual_a,
                virtual_b,
                pair_matrix,
                exchange_matrix,
                structure,
            )
        )
    return expanded


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
    wall_time_limit: float = 60.0,
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
        return value

    reference_x = np.zeros_like(x0)
    current_x = np.array(reference_x, copy=True)
    current_energy = evaluate(reference_x)

    trial_starts: list[np.ndarray] = []
    if np.any(np.abs(x0) > 0.0):
        trial_starts.append(np.array(x0, copy=True))
        trial_starts.append(np.array(-x0, copy=True))
    if cfg.initial_parameters and cfg.init_scale > 0.0:
        trial_starts.append(np.array(x0 + cfg.init_scale * rng.standard_normal(len(x0)), copy=True))

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
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the auxiliary N2_60 CUDA-Q VQE benchmark.")
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
