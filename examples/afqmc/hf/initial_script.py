"""
Minimal molecular AFQMC benchmark for external autoresearch agents.

This benchmark keeps the method family bounded to PySCF mean-field trials plus
an ipie AFQMC propagation stage under a fixed production wall-time budget.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ipie.analysis.extraction import extract_observable
from ipie.qmc.afqmc import AFQMC
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from pyscf import gto, scf

SCRIPT_DIR = Path(__file__).resolve().parent
AFQMC_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "model_registry.py").exists() else SCRIPT_DIR.parent
if str(AFQMC_ROOT) not in sys.path:
    sys.path.insert(0, str(AFQMC_ROOT))

from model_registry import MOLECULE_SPECS
from reference_energies import reference_energy


CHEMICAL_ACCURACY = 1e-3
MOLECULE_NAME = "HF"
SUPPORTED_MOLECULES = (MOLECULE_NAME,)


@dataclass(frozen=True)
class RunConfig:
    name: str
    trial: str
    orbital_basis: str
    chol_cut: float
    timestep: float
    num_walkers: int
    steps_per_block: int
    stabilize_freq: int
    pop_control_freq: int
    max_blocks_cap: int
    warmup_blocks: int
    pilot_blocks: int
    production_block_scale: float
    equilibration_fraction: float
    equilibration_min_blocks: int
    equilibration_max_blocks: int
    wall_buffer_seconds: float
    seed: int = 42


DEFAULT_CONFIG = RunConfig(
    name="simple_afqmc",
    trial="rhf",
    orbital_basis="mo",
    chol_cut=1e-5,
    timestep=0.005,
    num_walkers=96,
    steps_per_block=5,
    stabilize_freq=5,
    pop_control_freq=5,
    max_blocks_cap=2048,
    warmup_blocks=2,
    pilot_blocks=8,
    production_block_scale=0.6195019531,
    equilibration_fraction=0.08,
    equilibration_min_blocks=5,
    equilibration_max_blocks=96,
    wall_buffer_seconds=0.1,
    seed=42,
)


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def molecule_slug(name: str) -> str:
    return name.lower().replace("+", "_plus")


def build_problem(molecule_name: str = MOLECULE_NAME) -> dict:
    if molecule_name != MOLECULE_NAME:
        raise ValueError(f"unsupported molecule {molecule_name!r}; supported values: {SUPPORTED_MOLECULES}")
    spec = MOLECULE_SPECS[molecule_name]
    return {
        "name": spec.name,
        "basis": spec.basis,
        "charge": spec.charge,
        "spin": spec.spin,
        "unit": spec.unit,
        "atom": spec.atom,
    }


def basis_flag(cfg: RunConfig) -> bool:
    if cfg.orbital_basis == "mo":
        return False
    if cfg.orbital_basis == "ortho_ao":
        return True
    raise ValueError(f"unsupported orbital_basis {cfg.orbital_basis!r}")


def trial_solver(cfg: RunConfig, mol):
    if cfg.trial == "rhf":
        return scf.RHF(mol)
    if cfg.trial == "uhf":
        return scf.UHF(mol)
    raise ValueError(f"unsupported trial {cfg.trial!r}")


def cache_key(cfg: RunConfig, problem: dict) -> str:
    payload = {
        "molecule": problem["name"],
        "basis": problem["basis"],
        "charge": problem["charge"],
        "spin": problem["spin"],
        "trial": cfg.trial,
        "orbital_basis": cfg.orbital_basis,
        "chol_cut": cfg.chol_cut,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_dir(cfg: RunConfig, problem: dict) -> Path:
    return AFQMC_ROOT / "configs" / molecule_slug(problem["name"]) / cache_key(cfg, problem)


def prepare_ipie_inputs(cfg: RunConfig, problem: dict) -> dict:
    root = cache_dir(cfg, problem)
    root.mkdir(parents=True, exist_ok=True)
    chk_file = root / "scf.chk"
    ham_file = root / "hamiltonian.h5"
    wfn_file = root / "wavefunction.h5"
    meta_file = root / "prep.json"
    if chk_file.exists() and ham_file.exists() and wfn_file.exists() and meta_file.exists():
        payload = json.loads(meta_file.read_text())
        payload["chk_file"] = str(chk_file)
        payload["ham_file"] = str(ham_file)
        payload["wfn_file"] = str(wfn_file)
        return payload

    mol = gto.M(
        atom=problem["atom"],
        basis=problem["basis"],
        charge=problem["charge"],
        spin=problem["spin"],
        unit=problem["unit"],
        verbose=0,
    )
    mf = trial_solver(cfg, mol)
    mf.chkfile = str(chk_file)
    scf_energy = float(mf.kernel())
    if not getattr(mf, "converged", True):
        raise RuntimeError(f"{cfg.trial} did not converge for {problem['name']}")

    with contextlib.redirect_stdout(io.StringIO()):
        gen_ipie_input_from_pyscf_chk(
            str(chk_file),
            hamil_file=str(ham_file),
            wfn_file=str(wfn_file),
            verbose=False,
            chol_cut=cfg.chol_cut,
            ortho_ao=basis_flag(cfg),
        )

    payload = {
        "num_elec": [int(mol.nelec[0]), int(mol.nelec[1])],
        "scf_energy": scf_energy,
        "chk_file": str(chk_file),
        "ham_file": str(ham_file),
        "wfn_file": str(wfn_file),
        "basis": problem["basis"],
        "trial": cfg.trial,
        "orbital_basis": cfg.orbital_basis,
        "chol_cut": cfg.chol_cut,
    }
    meta_file.write_text(json.dumps(payload, indent=2))
    return payload


def run_driver(cfg: RunConfig, prepared: dict, num_blocks: int, estimator_file: Path) -> tuple[float, str]:
    capture = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(capture):
        afqmc = AFQMC.build_from_hdf5(
            tuple(int(value) for value in prepared["num_elec"]),
            prepared["ham_file"],
            prepared["wfn_file"],
            num_walkers=cfg.num_walkers,
            seed=cfg.seed,
            num_steps_per_block=cfg.steps_per_block,
            num_blocks=num_blocks,
            timestep=cfg.timestep,
            stabilize_freq=cfg.stabilize_freq,
            pop_control_freq=cfg.pop_control_freq,
            verbose=False,
        )
        afqmc.run(estimator_filename=str(estimator_file), verbose=False)
    return time.perf_counter() - start, capture.getvalue()


def energy_history(estimator_file: Path, target_energy: float) -> tuple[list[tuple[int, float, float]], float]:
    frame = extract_observable(str(estimator_file), "energy")
    energies = [float(value) for value in frame["ETotal"].tolist()]
    if not energies:
        raise RuntimeError("AFQMC estimator file contained no energy samples")
    history = [(idx, energy, energy - target_energy) for idx, energy in enumerate(energies, start=1)]
    return history, energies[-1]


def equilibration_blocks(cfg: RunConfig, history_length: int) -> int:
    scaled = int(history_length * cfg.equilibration_fraction)
    return min(history_length - 1, max(0, min(cfg.equilibration_max_blocks, max(cfg.equilibration_min_blocks, scaled))))


def averaged_production_energy(cfg: RunConfig, history: list[tuple[int, float, float]]) -> tuple[float, int]:
    burn = equilibration_blocks(cfg, len(history))
    production = history[burn:]
    return sum(energy for _step, energy, _error in production) / len(production), burn


def run_config(cfg: RunConfig, problem: dict, wall_time_limit: float, target_energy: float) -> dict:
    total_start = time.perf_counter()
    prepared = prepare_ipie_inputs(cfg, problem)
    prep_wall = time.perf_counter() - total_start

    with tempfile.TemporaryDirectory(prefix="afqmc_run_", dir=AFQMC_ROOT) as tmpdir_text:
        tmpdir = Path(tmpdir_text)
        warmup_file = tmpdir / "warmup.h5"
        warmup_blocks = max(1, min(cfg.warmup_blocks, cfg.max_blocks_cap))
        warmup_wall, _warmup_log = run_driver(cfg, prepared, warmup_blocks, warmup_file)

        pilot_file = tmpdir / "pilot.h5"
        pilot_blocks = max(1, min(cfg.pilot_blocks, cfg.max_blocks_cap))
        pilot_wall, pilot_log = run_driver(cfg, prepared, pilot_blocks, pilot_file)
        block_seconds = max((pilot_wall / pilot_blocks) * cfg.production_block_scale, 1e-3)

        remaining_budget = max(0.25, wall_time_limit - prep_wall - warmup_wall - pilot_wall - cfg.wall_buffer_seconds)
        production_blocks = max(1, min(cfg.max_blocks_cap, int(remaining_budget / block_seconds)))
        production_file = tmpdir / "production.h5"
        production_wall, production_log = run_driver(cfg, prepared, production_blocks, production_file)
        history, raw_last_energy = energy_history(production_file, target_energy)
        final_energy, equilibration = averaged_production_energy(cfg, history)

    final_error = final_energy - target_energy
    abs_final_error = abs(final_error)
    chem_acc_step = next((step for step, _energy, error in history if abs(error) <= CHEMICAL_ACCURACY), None)
    initial_energy = history[0][1]
    return {
        "config": config_to_dict(cfg),
        "molecule": problem["name"],
        "basis": problem["basis"],
        "metric": "abs_final_error",
        "lower_is_better": True,
        "score": abs_final_error,
        "chemical_accuracy": CHEMICAL_ACCURACY,
        "target_energy": target_energy,
        "reference_energy": target_energy,
        "trial_energy": prepared["scf_energy"],
        "final_energy": final_energy,
        "raw_last_energy": raw_last_energy,
        "final_error": final_error,
        "abs_final_error": abs_final_error,
        "energy_drop": initial_energy - final_energy,
        "chem_acc_step": chem_acc_step,
        "num_blocks": production_blocks,
        "warmup_blocks": warmup_blocks,
        "pilot_blocks": pilot_blocks,
        "equilibration_blocks": equilibration,
        "pilot_block_seconds": block_seconds,
        "wall_seconds": production_wall,
        "warmup_wall_seconds": warmup_wall,
        "prep_wall_seconds": prep_wall,
        "total_wall_seconds": time.perf_counter() - total_start,
        "wall_budget_seconds": wall_time_limit,
        "history": history,
        "pilot_log_lines": [line for line in pilot_log.strip().splitlines() if line.strip()][-8:],
        "production_log_lines": [line for line in production_log.strip().splitlines() if line.strip()][-8:],
        "supported_molecules": list(SUPPORTED_MOLECULES),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the fixed AFQMC molecular benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    args = parser.parse_args()

    problem = build_problem(MOLECULE_NAME)
    target = reference_energy(MOLECULE_NAME)
    if target is None:
        raise SystemExit(f"missing reference energy for {MOLECULE_NAME}; run compute_reference_energies.py first")

    result = run_config(DEFAULT_CONFIG, problem, wall_time_limit=args.wall_seconds, target_energy=target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
