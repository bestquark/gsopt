from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_registry import ACTIVE_MODELS
from simple_tn import RunConfig, build_problem, run_config

OUTPUT_PATH = Path(__file__).resolve().parent / "reference_energies.json"

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
}

REFERENCE_WALL_SECONDS = {
    "heisenberg_xxx_384": 600.0,
    "xxz_gapless_256": 480.0,
    "spin1_heisenberg_64": 420.0,
    "tfim_2d_4x4": 420.0,
    "heisenberg_2d_4x4": 600.0,
}


def compute_reference(model: str) -> dict:
    cfg = REFERENCE_CONFIGS[model]
    problem = build_problem(model)
    result = run_config(cfg, problem, wall_time_limit=REFERENCE_WALL_SECONDS[model])
    return {
        "reference_energy": result["final_energy"],
        "reference_energy_per_site": result["energy_per_site"],
        "nsites": result["nsites"],
        "shape": result["shape"],
        "wall_seconds": result["wall_seconds"],
        "max_bond_realized": result["max_bond_realized"],
        "history": result["history"],
        "config": result["config"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute frozen high-accuracy TN reference energies.")
    parser.add_argument("--model", choices=ACTIVE_MODELS)
    args = parser.parse_args()

    models = [args.model] if args.model else list(ACTIVE_MODELS)
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
