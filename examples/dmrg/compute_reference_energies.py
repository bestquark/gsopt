from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import qutip
import quimb.tensor as qtn

from model_registry import ACTIVE_MODELS, MODEL_SPECS

OUTPUT_PATH = Path(__file__).resolve().parent / "reference_energies.json"

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=UserWarning)

REFERENCE_CONFIGS = {
    "heisenberg_xxx_384": {
        "bond_schedule": (64, 96, 128, 160, 192, 256),
        "stage_sweeps": (2, 2, 2, 2, 2, 2),
        "cutoff": 1e-12,
        "solver_tol": 1e-8,
        "sweep_sequence": "RLRL",
        "init_bond_dim": 16,
        "init_seed": 123,
    },
    "xxz_gapless_256": {
        "bond_schedule": (64, 96, 128, 160, 192, 256),
        "stage_sweeps": (2, 2, 2, 2, 2, 2),
        "cutoff": 1e-12,
        "solver_tol": 1e-8,
        "sweep_sequence": "RLRL",
        "init_bond_dim": 16,
        "init_seed": 123,
    },
    "tfim_longitudinal_256": {
        "bond_schedule": (48, 64, 96, 128, 160, 192),
        "stage_sweeps": (2, 2, 2, 2, 2, 2),
        "cutoff": 1e-12,
        "solver_tol": 1e-8,
        "sweep_sequence": "RLRL",
        "init_bond_dim": 16,
        "init_seed": 123,
    },
    "spin1_heisenberg_64": {
        "bond_schedule": (32, 48, 64, 80, 96, 128),
        "stage_sweeps": (2, 2, 2, 2, 2, 2),
        "cutoff": 1e-12,
        "solver_tol": 1e-8,
        "sweep_sequence": "RLRL",
        "init_bond_dim": 8,
        "init_seed": 123,
    },
    "spin1_single_ion_critical_64": {
        "bond_schedule": (32, 48, 64, 80, 96, 128),
        "stage_sweeps": (2, 2, 2, 2, 2, 2),
        "cutoff": 1e-12,
        "solver_tol": 1e-8,
        "sweep_sequence": "RLRL",
        "init_bond_dim": 8,
        "init_seed": 123,
    },
}


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


def build_mpo(model: str):
    spec = MODEL_SPECS[model]
    arrays = spin_ops_arrays(spec.spin)
    builder = qtn.SpinHam1D(S=spec.spin, cyclic=spec.cyclic)
    for coeff, op_name in spec.onsite_terms:
        builder += (coeff, arrays[op_name])
    for coeff, left_name, right_name in spec.bond_terms:
        builder += (coeff, arrays[left_name], arrays[right_name])
    return builder.build_mpo(spec.chain_length)


def compute_reference(model: str) -> dict:
    spec = MODEL_SPECS[model]
    cfg = REFERENCE_CONFIGS[model]
    mpo = build_mpo(model)
    start = time.perf_counter()
    p0 = qtn.MPS_rand_state(
        spec.chain_length,
        bond_dim=cfg["init_bond_dim"],
        phys_dim=int(2 * spec.spin + 1),
        dtype="complex128",
        seed=cfg["init_seed"],
        cyclic=spec.cyclic,
    )
    solver = qtn.DMRG2(mpo, bond_dims=[cfg["bond_schedule"][0]], cutoffs=cfg["cutoff"], p0=p0)
    solver.opts["local_eig_tol"] = cfg["solver_tol"]
    solver.opts["local_eig_ncv"] = 8
    history: list[float] = []
    previous_direction = None
    directions = tuple(cfg["sweep_sequence"])
    direction_idx = 0
    for max_bond, sweeps in zip(cfg["bond_schedule"], cfg["stage_sweeps"]):
        for _ in range(sweeps):
            direction = directions[direction_idx % len(directions)]
            direction_idx += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                energy = solver.sweep(
                    direction=direction,
                    canonize=previous_direction in (None, direction),
                    max_bond=max_bond,
                    cutoff=cfg["cutoff"],
                    cutoff_mode=solver.opts["bond_compress_cutoff_mode"],
                    method=solver.opts["bond_compress_method"],
                    verbosity=0,
                )
            energy = float(np.real(energy))
            solver.energies.append(energy)
            history.append(energy)
            previous_direction = direction
    reference_energy = float(np.real(solver.energy))
    return {
        "reference_energy": reference_energy,
        "reference_energy_per_site": reference_energy / spec.chain_length,
        "chain_length": spec.chain_length,
        "cyclic": spec.cyclic,
        "wall_seconds": time.perf_counter() - start,
        "max_bond_realized": int(solver.state.max_bond()),
        "history": history,
        "config": {
            "bond_schedule": list(cfg["bond_schedule"]),
            "stage_sweeps": list(cfg["stage_sweeps"]),
            "cutoff": cfg["cutoff"],
            "solver_tol": cfg["solver_tol"],
            "sweep_sequence": cfg["sweep_sequence"],
            "init_bond_dim": cfg["init_bond_dim"],
            "init_seed": cfg["init_seed"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute frozen high-accuracy DMRG reference energies.")
    parser.add_argument("--model", choices=ACTIVE_MODELS)
    args = parser.parse_args()

    models = [args.model] if args.model else list(ACTIVE_MODELS)
    existing = {}
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text())
    for model in models:
        print(f"computing reference for {model}...", flush=True)
        existing[model] = compute_reference(model)
    OUTPUT_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps({"reference_file": str(OUTPUT_PATH), "models": models}, indent=2))


if __name__ == "__main__":
    main()
