"""
Minimal CUDA-Q VQE benchmark for the fixed BH external-agent lane.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cudaq
import numpy as np
from cudaq import spin
from openfermion import MolecularData, get_fermion_operator, get_sparse_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from scipy.optimize import minimize

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from config_override import load_dataclass_override

cudaq.set_target("qpp-cpu")

CHEMICAL_ACCURACY = 1e-3
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_VQE_CONFIG_JSON"


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
    ansatz: str
    layers: int
    optimizer: str
    max_steps: int
    init_scale: float = 0.1
    seed: int = 42
    initial_parameters: tuple[float, ...] = ()
    cobyla_rhobeg: float = 0.5
    cobyla_tol: float = 1e-4
    powell_xtol: float = 1e-3
    powell_ftol: float = 1e-5
    nelder_mead_xatol: float = 1e-3
    nelder_mead_fatol: float = 1e-5


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
MOLECULE_NAME = "BH"
SUPPORTED_MOLECULES = (MOLECULE_NAME,)

DEFAULT_CONFIG = RunConfig(
    name="simple_vqe",
    ansatz="uccsd",
    layers=0,
    optimizer="cobyla",
    max_steps=4096,
    init_scale=0.0,
    seed=29,
    initial_parameters=(
        0.00010520257969425875,
        0.00023916513811494088,
        -6.75738662583263e-05,
        -0.00015355097125414614,
        0.35378702633408265,
        0.00010321404358987446,
        -2.346035861745284e-05,
        0.35938063285316296,
    ),
    cobyla_rhobeg=5e-5,
    cobyla_tol=1e-12,
    powell_xtol=1e-8,
    powell_ftol=1e-10,
    nelder_mead_xatol=1e-3,
    nelder_mead_fatol=1e-5,
)


@cudaq.kernel
def hea_ry_ring_kernel(n_qubits: int, layers: int, active_electrons: int, theta: list[float]):
    qubits = cudaq.qvector(n_qubits)
    for idx in range(active_electrons):
        x(qubits[idx])
    cursor = 0
    for _ in range(layers):
        for qubit in range(n_qubits):
            ry(theta[cursor], qubits[qubit])
            cursor += 1
        for qubit in range(n_qubits - 1):
            x.ctrl(qubits[qubit], qubits[qubit + 1])
        if n_qubits > 2:
            x.ctrl(qubits[n_qubits - 1], qubits[0])


@cudaq.kernel
def hea_ryrz_ring_kernel(n_qubits: int, layers: int, active_electrons: int, theta: list[float]):
    qubits = cudaq.qvector(n_qubits)
    for idx in range(active_electrons):
        x(qubits[idx])
    cursor = 0
    for _ in range(layers):
        for qubit in range(n_qubits):
            ry(theta[cursor], qubits[qubit])
            cursor += 1
            rz(theta[cursor], qubits[qubit])
            cursor += 1
        for qubit in range(n_qubits - 1):
            x.ctrl(qubits[qubit], qubits[qubit + 1])
        if n_qubits > 2:
            x.ctrl(qubits[n_qubits - 1], qubits[0])


@cudaq.kernel
def uccsd_kernel(n_qubits: int, layers: int, active_electrons: int, theta: list[float]):
    qubits = cudaq.qvector(n_qubits)
    for idx in range(active_electrons):
        x(qubits[idx])
    cudaq.kernels.uccsd(qubits, theta, active_electrons, n_qubits)


class WallTimeReached(RuntimeError):
    pass


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def config_from_dict(data: dict) -> RunConfig:
    return RunConfig(**data)


def runtime_config() -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, DEFAULT_CONFIG, RunConfig)


def config_signature(cfg: RunConfig) -> tuple:
    return (
        cfg.ansatz,
        cfg.layers,
        cfg.optimizer,
        cfg.max_steps,
        cfg.init_scale,
        cfg.seed,
        tuple(cfg.initial_parameters),
        cfg.cobyla_rhobeg,
        cfg.cobyla_tol,
        cfg.powell_xtol,
        cfg.powell_ftol,
        cfg.nelder_mead_xatol,
        cfg.nelder_mead_fatol,
    )


def make_config_name(cfg: RunConfig) -> str:
    init_part = str(cfg.init_scale).replace(".", "p")
    return f"{cfg.ansatz}-L{cfg.layers}-{cfg.optimizer}-steps{cfg.max_steps}-init{init_part}-seed{cfg.seed}"


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


def estimate_circuit_depth(cfg: RunConfig, problem: dict) -> int:
    if cfg.ansatz == "hea_ry_ring":
        return 2 * cfg.layers
    if cfg.ansatz == "hea_ryrz_ring":
        return 3 * cfg.layers
    if cfg.ansatz == "uccsd":
        return 2 * cudaq.kernels.uccsd_num_parameters(problem["active_electrons"], problem["n_qubits"])
    raise ValueError(f"unknown ansatz: {cfg.ansatz}")


def num_params(cfg: RunConfig, n_qubits: int, active_electrons: int) -> int:
    if cfg.ansatz == "hea_ry_ring":
        return cfg.layers * n_qubits
    if cfg.ansatz == "hea_ryrz_ring":
        return cfg.layers * 2 * n_qubits
    if cfg.ansatz == "uccsd":
        return cudaq.kernels.uccsd_num_parameters(active_electrons, n_qubits)
    raise ValueError(f"unknown ansatz: {cfg.ansatz}")


def kernel_for_ansatz(name: str):
    if name == "hea_ry_ring":
        return hea_ry_ring_kernel
    if name == "hea_ryrz_ring":
        return hea_ryrz_ring_kernel
    if name == "uccsd":
        return uccsd_kernel
    raise ValueError(f"unknown ansatz: {name}")


def energy_from_params(params: np.ndarray, cfg: RunConfig, problem: dict) -> float:
    kernel = kernel_for_ansatz(cfg.ansatz)
    parameter_list = [float(value) for value in np.asarray(params, dtype=np.float64)]
    result = cudaq.observe(
        kernel,
        problem["hamiltonian"],
        problem["n_qubits"],
        cfg.layers,
        problem["active_electrons"],
        parameter_list,
    )
    return float(result.expectation())


def make_initial_parameters(cfg: RunConfig, problem: dict, rng: np.random.Generator) -> np.ndarray:
    n_parameters = num_params(cfg, problem["n_qubits"], problem["active_electrons"])
    if cfg.initial_parameters:
        x0 = np.asarray(cfg.initial_parameters, dtype=np.float64)
        if x0.size != n_parameters:
            raise ValueError(
                f"initial_parameters has length {x0.size}, expected {n_parameters} for ansatz {cfg.ansatz}"
            )
        if cfg.init_scale > 0.0:
            x0 = x0 + cfg.init_scale * rng.standard_normal(n_parameters)
        return x0
    if cfg.init_scale == 0.0:
        return np.zeros(n_parameters, dtype=np.float64)
    return cfg.init_scale * rng.standard_normal(n_parameters)


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
    last_x: np.ndarray | None = None
    last_value: float | None = None

    def timed_out() -> bool:
        return (time.perf_counter() - start) >= wall_time_limit

    def evaluate(x: np.ndarray) -> float:
        nonlocal nfev, best_energy, best_x, last_x, last_value
        if timed_out():
            raise WallTimeReached
        trial_x = np.asarray(x, dtype=np.float64)
        if last_x is not None and np.array_equal(trial_x, last_x):
            value = last_value
        else:
            value = energy_from_params(trial_x, cfg, problem)
            last_x = trial_x.copy()
            last_value = value
            nfev += 1
        if value < best_energy:
            best_energy = value
            best_x = trial_x.copy()
        return float(value)

    best_energy = float("inf")
    best_x = np.array(x0, copy=True)
    best_energy = evaluate(x0)
    history.append((1, best_energy, best_energy - target_energy))
    if best_energy - target_energy <= chemical_accuracy:
        chem_acc_step = 1

    def objective(x: np.ndarray) -> float:
        return evaluate(x)

    def callback(xk: np.ndarray):
        nonlocal chem_acc_step
        if timed_out():
            raise WallTimeReached
        value = evaluate(xk)
        error = value - target_energy
        step = len(history) + 1
        history.append((step, value, error))
        if chem_acc_step is None and error <= chemical_accuracy:
            chem_acc_step = step
        if timed_out():
            raise WallTimeReached

    method_name = {
        "cobyla": "COBYLA",
        "powell": "Powell",
        "nelder-mead": "Nelder-Mead",
    }.get(cfg.optimizer)
    if method_name is None:
        raise ValueError(f"unknown optimizer: {cfg.optimizer}")

    result_x = x0
    result_fun = best_energy
    try:
        result = minimize(
            objective,
            x0=x0,
            method=method_name,
            callback=callback,
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
        "circuit_depth": estimate_circuit_depth(cfg, problem),
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
    parser = argparse.ArgumentParser(description="Run the fixed BH CUDA-Q VQE benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=5.0)
    args = parser.parse_args()

    problem = build_problem(MOLECULE_NAME)
    result = run_config(runtime_config(), problem, chemical_accuracy=CHEMICAL_ACCURACY, wall_time_limit=args.wall_seconds)
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
