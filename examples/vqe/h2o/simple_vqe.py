"""
Minimal CUDA-Q VQE benchmark for the fixed H2O external-agent lane.
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
    init_scale: float = 0.2
    seed: int = 42
    step_size: float = 0.2
    min_step_size: float = 1e-3
    cobyla_rhobeg: float = 0.1
    cobyla_tol: float = 1e-5
    powell_xtol: float = 1e-4
    powell_ftol: float = 1e-6
    nelder_mead_xatol: float = 1e-4
    nelder_mead_fatol: float = 1e-6


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
MOLECULE_NAME = "H2O"
SUPPORTED_MOLECULES = (MOLECULE_NAME,)

DEFAULT_CONFIG = RunConfig(
    name="simple_vqe",
    param_model="uccsd_full",
    optimizer="cobyla",
    max_steps=4096,
    init_scale=0.000625,
    seed=257,
    step_size=0.18,
    min_step_size=1e-3,
    cobyla_rhobeg=0.00036808221403564824,
    cobyla_tol=4.05424458014887e-12,
    powell_xtol=0.017721251564583946,
    powell_ftol=1.0207220225422267e-14,
    nelder_mead_xatol=2.7600279500135757e-08,
    nelder_mead_fatol=1.6013042113252859e-07,
)

REFERENCE_SPIN_PAIRED_SYMMETRIC = np.asarray(
    [
        0.004333197062818001,
        0.006475696737450206,
        -0.005714577620291297,
        0.08690958461668224,
        0.08376250070042049,
        0.05829507543504292,
        -0.0008681303177481437,
        0.000766160085669451,
        0.002013180834907711,
    ],
    dtype=np.float64,
)

REFERENCE_SPIN_PAIRED_FULL_DOUBLES = np.asarray(
    [
        -0.002243114831027941,
        0.007502925221260322,
        -0.0032390350103357595,
        0.08935144572005857,
        0.005531221072339278,
        -0.0017961058197368059,
        0.0041683084950668425,
        0.09084096210858049,
        0.0005939937647919922,
        0.002254692750372833,
        -0.0001359641001935978,
        0.06173479421240287,
    ],
    dtype=np.float64,
)

REFERENCE_FULL_UCCSD = np.asarray(
    [
        -0.0005776854818457494,
        0.007243118547180653,
        -0.002162854474617421,
        0.0007717490141278833,
        0.005352302486079926,
        0.00045731707010367433,
        0.086779713260905,
        0.001652851615888544,
        0.0004779799976069199,
        0.0020398928641036976,
        0.09275430736414024,
        0.0012490637203208723,
        0.00019093127950681947,
        0.001175549393177736,
        0.05837168574106286,
    ],
    dtype=np.float64,
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


def estimate_circuit_depth(_cfg: RunConfig) -> int:
    return 1


def tied_uccsd_counts(active_electrons: int, n_qubits: int) -> tuple[int, int]:
    singles = active_electrons * (n_qubits - active_electrons) // 2
    total = int(cudaq.kernels.uccsd_num_parameters(active_electrons, n_qubits))
    return singles, total - singles


def num_params(cfg: RunConfig, problem: dict) -> int:
    if cfg.param_model == "uccsd_full":
        return int(cudaq.kernels.uccsd_num_parameters(problem["active_electrons"], problem["n_qubits"]))
    if cfg.param_model == "uccsd_tied_sd":
        return 2
    if cfg.param_model == "uccsd_doubles_triplet":
        return 3
    if cfg.param_model == "uccsd_sd_triplet":
        return 4
    if cfg.param_model == "uccsd_spin_paired_symmetric":
        return 9
    if cfg.param_model == "uccsd_spin_paired_full_doubles":
        return 12
    raise ValueError(f"unknown parameter model: {cfg.param_model}")


def expand_params(params: np.ndarray, cfg: RunConfig, problem: dict) -> list[float]:
    raw_params = np.asarray(params, dtype=np.float64)
    if cfg.param_model == "uccsd_full":
        return [float(value) for value in raw_params]

    singles, doubles = tied_uccsd_counts(problem["active_electrons"], problem["n_qubits"])
    if cfg.param_model == "uccsd_tied_sd":
        single_angle = float(raw_params[0])
        double_angle = float(raw_params[1])
        return [single_angle] * singles + [double_angle] * doubles
    if cfg.param_model == "uccsd_doubles_triplet":
        angles = [float(raw_params[0]), float(raw_params[1]), float(raw_params[2])]
        return [0.0] * singles + [angles[index % 3] for index in range(doubles)]
    if cfg.param_model == "uccsd_sd_triplet":
        single_angle = float(raw_params[0])
        angles = [float(raw_params[1]), float(raw_params[2]), float(raw_params[3])]
        return [single_angle] * singles + [angles[index % 3] for index in range(doubles)]
    if cfg.param_model == "uccsd_spin_paired_symmetric":
        single_angles = [float(raw_params[0]), float(raw_params[1]), float(raw_params[2])]
        diagonal_doubles = [float(raw_params[3]), float(raw_params[4]), float(raw_params[5])]
        off_diagonal_doubles = {
            (0, 1): float(raw_params[6]),
            (0, 2): float(raw_params[7]),
            (1, 2): float(raw_params[8]),
        }
        single_params = single_angles + single_angles
        double_params: list[float] = []
        for alpha_index in range(3):
            for beta_index in range(3):
                if alpha_index == beta_index:
                    double_params.append(diagonal_doubles[alpha_index])
                else:
                    key = tuple(sorted((alpha_index, beta_index)))
                    double_params.append(off_diagonal_doubles[key])
        return single_params + double_params
    if cfg.param_model == "uccsd_spin_paired_full_doubles":
        single_angles = [float(raw_params[0]), float(raw_params[1]), float(raw_params[2])]
        single_params = single_angles + single_angles
        double_params = [float(value) for value in raw_params[3:12]]
        return single_params + double_params
    raise ValueError(f"unknown parameter model: {cfg.param_model}")


def warm_start_params(cfg: RunConfig) -> np.ndarray | None:
    if cfg.param_model == "uccsd_spin_paired_symmetric":
        return np.array(REFERENCE_SPIN_PAIRED_SYMMETRIC, copy=True)
    if cfg.param_model == "uccsd_spin_paired_full_doubles":
        return np.array(REFERENCE_SPIN_PAIRED_FULL_DOUBLES, copy=True)
    if cfg.param_model == "uccsd_full":
        return np.array(REFERENCE_FULL_UCCSD, copy=True)
    return None


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
    if cfg.optimizer == "nelder-mead":
        return {
            "maxiter": cfg.max_steps,
            "disp": False,
            "xatol": cfg.nelder_mead_xatol,
            "fatol": cfg.nelder_mead_fatol,
            "adaptive": True,
        }
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def run_coordinate_search(
    cfg: RunConfig,
    problem: dict,
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
    warm_start = warm_start_params(cfg)
    if warm_start is None:
        x0 = cfg.init_scale * rng.standard_normal(num_params(cfg, problem))
    else:
        x0 = np.array(warm_start, copy=True)

    history: list[tuple[int, float, float]] = []
    chem_acc_step = None
    nfev = 0
    start = time.perf_counter()
    best_energy = float("inf")
    best_x = np.zeros_like(x0)

    def timed_out() -> bool:
        return (time.perf_counter() - start) >= wall_time_limit

    def evaluate(x: np.ndarray) -> float:
        nonlocal nfev, best_energy, best_x, chem_acc_step
        if timed_out():
            raise WallTimeReached
        value = energy_from_params(x, cfg, problem)
        nfev += 1
        error = value - target_energy
        history.append((len(history) + 1, value, error))
        if chem_acc_step is None and error <= chemical_accuracy:
            chem_acc_step = len(history)
        if value < best_energy:
            best_energy = value
            best_x = np.asarray(x, dtype=np.float64).copy()
        return value

    reference_x = np.zeros_like(x0)
    current_x = np.array(reference_x, copy=True)
    current_energy = evaluate(reference_x)

    if cfg.init_scale > 0.0:
        if warm_start is None:
            trial_starts = (x0, -x0)
        else:
            perturbation = cfg.init_scale * rng.standard_normal(len(x0))
            trial_starts = (x0, x0 + perturbation, x0 - perturbation)
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
                problem=problem,
                current_x=current_x,
                current_energy=current_energy,
                evaluate=evaluate,
                timed_out=timed_out,
            )
        else:
            method_name = {
                "cobyla": "COBYLA",
                "powell": "Powell",
                "nelder-mead": "Nelder-Mead",
            }.get(cfg.optimizer)
            if method_name is None:
                raise ValueError(f"unknown optimizer: {cfg.optimizer}")
            result = minimize(
                evaluate,
                x0=current_x,
                method=method_name,
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
    parser = argparse.ArgumentParser(description="Run the fixed H2O CUDA-Q VQE benchmark.")
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
