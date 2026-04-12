from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import quimb as qu

from model_registry import ACTIVE_MODELS, MUTUAL_INFO_MODELS, MODEL_SPECS
from simple_tn import (
    RunConfig,
    build_problem,
    build_sparse_1d_hamiltonian,
    mutual_information_matrix_from_mps_state,
    mutual_information_matrix_from_statevector,
    run_config,
)

OUTPUT_PATH = Path(__file__).resolve().parent / "reference_energies.json"
REFERENCE_MODELS = tuple(dict.fromkeys((*ACTIVE_MODELS, *MUTUAL_INFO_MODELS)))

REFERENCE_CONFIGS = {
    "heisenberg_xxx_384": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="random",
        bond_schedule=(64, 96, 128, 160, 192, 256),
        cutoff=1e-12,
        solver_tol=1e-8,
        max_sweeps=18,
        tau=0.05,
        chi=64,
        init_bond_dim=16,
        init_seed=123,
        local_eig_ncv=8,
    ),
    "xxz_gapless_256": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="random",
        bond_schedule=(48, 64, 96, 128, 160, 192),
        cutoff=1e-12,
        solver_tol=1e-8,
        max_sweeps=18,
        tau=0.05,
        chi=64,
        init_bond_dim=12,
        init_seed=123,
        local_eig_ncv=8,
    ),
    "spin1_heisenberg_64": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="neel",
        bond_schedule=(24, 32, 48, 64, 80, 96, 128),
        cutoff=1e-12,
        solver_tol=1e-8,
        max_sweeps=18,
        tau=0.05,
        chi=64,
        init_bond_dim=8,
        init_seed=123,
        local_eig_ncv=8,
    ),
    "tfim_2d_4x4": RunConfig(
        name="reference",
        method="tebd2d",
        init_state="plus",
        bond_schedule=(2, 3, 4, 5),
        cutoff=1e-9,
        solver_tol=1e-8,
        max_sweeps=16,
        tau=0.05,
        chi=48,
        init_bond_dim=2,
        init_seed=123,
        local_eig_ncv=8,
    ),
    "heisenberg_2d_4x4": RunConfig(
        name="reference",
        method="tebd2d",
        init_state="checkerboard",
        bond_schedule=(2, 3, 4),
        cutoff=1e-9,
        solver_tol=1e-8,
        max_sweeps=12,
        tau=0.04,
        chi=48,
        init_bond_dim=2,
        init_seed=123,
        local_eig_ncv=8,
    ),
    "heisenberg_xxx_64": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="neel",
        bond_schedule=(32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384),
        cutoff=1e-14,
        solver_tol=1e-11,
        max_sweeps=72,
        tau=0.05,
        chi=64,
        init_bond_dim=16,
        init_seed=123,
        local_eig_ncv=24,
    ),
    "xxz_gapless_64": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="neel",
        bond_schedule=(32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384),
        cutoff=1e-14,
        solver_tol=1e-11,
        max_sweeps=72,
        tau=0.05,
        chi=64,
        init_bond_dim=16,
        init_seed=123,
        local_eig_ncv=24,
    ),
    "tfim_critical_64": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="plus",
        bond_schedule=(16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256),
        cutoff=1e-14,
        solver_tol=1e-11,
        max_sweeps=64,
        tau=0.05,
        chi=64,
        init_bond_dim=12,
        init_seed=123,
        local_eig_ncv=24,
    ),
    "xx_critical_64": RunConfig(
        name="reference",
        method="dmrg2",
        init_state="neel",
        bond_schedule=(24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320),
        cutoff=1e-14,
        solver_tol=1e-11,
        max_sweeps=64,
        tau=0.05,
        chi=64,
        init_bond_dim=12,
        init_seed=123,
        local_eig_ncv=24,
    ),
}

REFERENCE_WALL_SECONDS = {
    "heisenberg_xxx_384": 600.0,
    "xxz_gapless_256": 480.0,
    "spin1_heisenberg_64": 420.0,
    "tfim_2d_4x4": 420.0,
    "heisenberg_2d_4x4": 600.0,
    "heisenberg_xxx_64": 3600.0,
    "xxz_gapless_64": 3600.0,
    "tfim_critical_64": 2400.0,
    "xx_critical_64": 2400.0,
}


def compute_exact_reference(model: str) -> dict:
    spec = MODEL_SPECS[model]
    if spec.geometry != "1d":
        raise ValueError(f"exact reference is only implemented for 1D models, got {model}")
    h_sparse = build_sparse_1d_hamiltonian(spec)
    ground_state = qu.groundstate(h_sparse)
    energy = float(np.real((qu.dag(ground_state) @ (h_sparse @ ground_state))[0, 0]))
    ground_state_vec = np.asarray(ground_state).reshape(-1)
    phys_dim = int(2 * spec.spin + 1)
    mutual_information = mutual_information_matrix_from_statevector(ground_state_vec, dims=(phys_dim,) * spec.nsites)
    return {
        "reference_energy": energy,
        "reference_energy_per_site": energy / spec.nsites,
        "nsites": spec.nsites,
        "shape": [spec.lx, spec.ly],
        "wall_seconds": None,
        "max_bond_realized": None,
        "history": [],
        "config": {"method": "exact_sparse_groundstate"},
        "reference_kind": "exact_sparse_groundstate",
        "mutual_information_matrix": mutual_information.tolist(),
    }


def compute_reference(model: str) -> dict:
    spec = MODEL_SPECS[model]
    if spec.geometry == "1d" and spec.nsites <= 20:
        return compute_exact_reference(model)
    cfg = REFERENCE_CONFIGS[model]
    problem = build_problem(model)
    result = run_config(cfg, problem, wall_time_limit=REFERENCE_WALL_SECONDS[model], return_state=True)
    state = result.pop("state_obj", None)
    mutual_information = None
    if spec.geometry == "1d" and state is not None:
        mutual_information = mutual_information_matrix_from_mps_state(state, spec.nsites).tolist()
    return {
        "reference_energy": result["final_energy"],
        "reference_energy_per_site": result["energy_per_site"],
        "nsites": result["nsites"],
        "shape": result["shape"],
        "wall_seconds": result["wall_seconds"],
        "max_bond_realized": result["max_bond_realized"],
        "history": result["history"],
        "config": result["config"],
        "reference_kind": "high_accuracy_run",
        "mutual_information_matrix": mutual_information,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute frozen high-accuracy TN reference energies.")
    parser.add_argument("--model", choices=REFERENCE_MODELS)
    args = parser.parse_args()

    models = [args.model] if args.model else list(REFERENCE_MODELS)
    existing = {}
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text().strip():
        existing = json.loads(OUTPUT_PATH.read_text())
    for model in models:
        print(f"computing reference for {model}...", flush=True)
        existing[model] = compute_reference(model)
        OUTPUT_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps({"reference_file": str(OUTPUT_PATH), "models": models}, indent=2))


if __name__ == "__main__":
    main()
