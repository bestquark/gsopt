"""
Molecular AFQMC benchmark for the AFQMC lane surface.

This lane uses PySCF to build molecular Hamiltonians and trial wavefunctions,
ipie to run real phaseless AFQMC, and offline CCSD(T) as the deterministic
reference method. The live optimization target is a risk-adjusted score
E + 2 * stderr computed from the final 40% tail of retained block energies.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyscf import cc, gto, scf

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_ROOT = SCRIPT_DIR.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from config_override import load_dataclass_override

try:
    from .model_registry import ACTIVE_SYSTEMS, SYSTEM_SPECS
    from .reference_energies import reference_energy
except ImportError:
    from model_registry import ACTIVE_SYSTEMS, SYSTEM_SPECS
    from reference_energies import reference_energy

CONFIG_OVERRIDE_ENV = "AUTORESEARCH_AFQMC_CONFIG_JSON"
ALLOWED_TRIALS = {"rhf", "uhf"}
ALLOWED_INIT_GUESSES = {"minao", "atom", "1e", "huckel", "mod_huckel"}
MIN_SCF_CONV_TOL = 1e-12
MAX_SCF_CONV_TOL = 1e-4
MAX_SCF_MAX_CYCLE = 256
MAX_RUNTIME_DIIS_SPACE = 20
MAX_RUNTIME_LEVEL_SHIFT = 1.0
MAX_RUNTIME_DAMPING = 0.5
MIN_CHOL_CUT = 1e-8
MAX_CHOL_CUT = 1e-4
MIN_TIMESTEP = 1e-3
MAX_TIMESTEP = 5e-2
MIN_STEPS_PER_BLOCK = 5
MAX_STEPS_PER_BLOCK = 100
MIN_NUM_BLOCKS = 20
MAX_NUM_BLOCKS = 200
MIN_WALKERS_PER_RANK = 8
MAX_WALKERS_PER_RANK = 512
MIN_FREQUENCY = 1
MAX_FREQUENCY = 50
MIN_PRODUCTION_BLOCKS = 10
SCORING_TAIL_FRACTION = 0.4
LIVE_OBJECTIVE_STDERR_WEIGHT = 2.0
LIVE_OBJECTIVE_METRIC = "tail_mean_plus_2stderr"
REFERENCE_CCSD_CONV_TOL = 1e-10
REFERENCE_CCSD_CONV_TOL_NORMT = 1e-8
REFERENCE_CCSD_MAX_CYCLE = 256


@dataclass(frozen=True)
class RunConfig:
    name: str
    trial: str
    scf_conv_tol: float
    scf_max_cycle: int
    diis_space: int
    level_shift: float
    damping: float
    init_guess: str
    chol_cut: float
    num_walkers_per_rank: int
    num_steps_per_block: int
    num_blocks: int
    timestep: float
    stabilize_freq: int
    pop_control_freq: int


REFERENCE_SCF_CONFIG = RunConfig(
    name="reference_rhf_ccsd_t",
    trial="rhf",
    scf_conv_tol=1e-10,
    scf_max_cycle=256,
    diis_space=12,
    level_shift=0.0,
    damping=0.0,
    init_guess="minao",
    chol_cut=1e-8,
    num_walkers_per_rank=128,
    num_steps_per_block=25,
    num_blocks=40,
    timestep=0.005,
    stabilize_freq=5,
    pop_control_freq=5,
)


def _import_ipie_modules():
    try:
        from ipie.analysis.extraction import extract_observable
        from ipie.qmc.afqmc import AFQMC
        from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
    except Exception as exc:  # pragma: no cover - cluster/runtime dependent
        raise RuntimeError(
            "ipie AFQMC requires a working ipie installation and MPI runtime. "
            "Run AFQMC evaluations on the configured cluster environment."
        ) from exc
    return AFQMC, extract_observable, gen_ipie_input_from_pyscf_chk


def _try_import_mpi_comm():
    try:
        from mpi4py import MPI
    except Exception:
        return None
    return MPI.COMM_WORLD


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def _literal_assignment_expr(tree: ast.Module, name: str):
    result = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    result = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            result = node.value
    return result


def _safe_eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval_node(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError("unary numeric operator applied to non-numeric value")
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RunConfig":
        if node.args:
            raise ValueError("RunConfig(...) must use keyword arguments only")
        payload = {keyword.arg: _safe_eval_node(keyword.value) for keyword in node.keywords if keyword.arg}
        return RunConfig(**payload)
    raise ValueError(f"unsupported expression in AFQMC source: {ast.dump(node, include_attributes=False)}")


def load_source_definition(source_file: Path) -> tuple[str, RunConfig]:
    tree = ast.parse(source_file.read_text(), filename=str(source_file))
    system_expr = _literal_assignment_expr(tree, "SYSTEM_NAME")
    config_expr = _literal_assignment_expr(tree, "DEFAULT_CONFIG")
    if system_expr is None:
        raise ValueError(f"{source_file} does not define SYSTEM_NAME")
    if config_expr is None:
        raise ValueError(f"{source_file} does not define DEFAULT_CONFIG")
    system_name = _safe_eval_node(system_expr)
    cfg = _safe_eval_node(config_expr)
    if not isinstance(system_name, str):
        raise ValueError("SYSTEM_NAME must be a string literal")
    if not isinstance(cfg, RunConfig):
        raise ValueError("DEFAULT_CONFIG must be a RunConfig(...) literal")
    return system_name, cfg


def validate_runtime_config(cfg: RunConfig):
    if cfg.trial not in ALLOWED_TRIALS:
        raise ValueError(f"unsupported trial {cfg.trial!r}")
    if not (MIN_SCF_CONV_TOL <= cfg.scf_conv_tol <= MAX_SCF_CONV_TOL):
        raise ValueError(f"scf_conv_tol must be between {MIN_SCF_CONV_TOL:g} and {MAX_SCF_CONV_TOL:g}")
    if not (1 <= int(cfg.scf_max_cycle) <= MAX_SCF_MAX_CYCLE):
        raise ValueError(f"scf_max_cycle must be between 1 and {MAX_SCF_MAX_CYCLE}")
    if not (1 <= int(cfg.diis_space) <= MAX_RUNTIME_DIIS_SPACE):
        raise ValueError(f"diis_space must be between 1 and {MAX_RUNTIME_DIIS_SPACE}")
    if not (0.0 <= float(cfg.level_shift) <= MAX_RUNTIME_LEVEL_SHIFT):
        raise ValueError(f"level_shift must be between 0 and {MAX_RUNTIME_LEVEL_SHIFT}")
    if not (0.0 <= float(cfg.damping) <= MAX_RUNTIME_DAMPING):
        raise ValueError(f"damping must be between 0 and {MAX_RUNTIME_DAMPING}")
    if cfg.init_guess not in ALLOWED_INIT_GUESSES:
        raise ValueError(f"unsupported init_guess {cfg.init_guess!r}")
    if not (MIN_CHOL_CUT <= float(cfg.chol_cut) <= MAX_CHOL_CUT):
        raise ValueError(f"chol_cut must be between {MIN_CHOL_CUT:g} and {MAX_CHOL_CUT:g}")
    if not (MIN_WALKERS_PER_RANK <= int(cfg.num_walkers_per_rank) <= MAX_WALKERS_PER_RANK):
        raise ValueError(f"num_walkers_per_rank must be between {MIN_WALKERS_PER_RANK} and {MAX_WALKERS_PER_RANK}")
    if not (MIN_STEPS_PER_BLOCK <= int(cfg.num_steps_per_block) <= MAX_STEPS_PER_BLOCK):
        raise ValueError(f"num_steps_per_block must be between {MIN_STEPS_PER_BLOCK} and {MAX_STEPS_PER_BLOCK}")
    if not (MIN_NUM_BLOCKS <= int(cfg.num_blocks) <= MAX_NUM_BLOCKS):
        raise ValueError(f"num_blocks must be between {MIN_NUM_BLOCKS} and {MAX_NUM_BLOCKS}")
    if not (MIN_TIMESTEP <= float(cfg.timestep) <= MAX_TIMESTEP):
        raise ValueError(f"timestep must be between {MIN_TIMESTEP:g} and {MAX_TIMESTEP:g}")
    if not (MIN_FREQUENCY <= int(cfg.stabilize_freq) <= MAX_FREQUENCY):
        raise ValueError(f"stabilize_freq must be between {MIN_FREQUENCY} and {MAX_FREQUENCY}")
    if not (MIN_FREQUENCY <= int(cfg.pop_control_freq) <= MAX_FREQUENCY):
        raise ValueError(f"pop_control_freq must be between {MIN_FREQUENCY} and {MAX_FREQUENCY}")


def build_molecule(spec):
    mol = gto.Mole()
    mol.atom = spec.atom
    mol.basis = spec.basis
    mol.unit = spec.unit
    mol.charge = spec.charge
    mol.spin = spec.spin
    mol.verbose = 0
    mol.build()
    return mol


def build_solver(mol, cfg: RunConfig):
    if cfg.trial == "rhf":
        solver = scf.RHF(mol)
    elif cfg.trial == "uhf":
        solver = scf.UHF(mol)
    else:
        raise ValueError(f"unsupported trial {cfg.trial!r}")
    solver.conv_tol = cfg.scf_conv_tol
    solver.max_cycle = cfg.scf_max_cycle
    solver.diis_space = cfg.diis_space
    solver.level_shift = cfg.level_shift
    solver.damp = cfg.damping
    solver.verbose = 0
    return solver


def build_problem(system_name: str) -> dict:
    if system_name not in ACTIVE_SYSTEMS:
        raise ValueError(f"unsupported molecular AFQMC system {system_name!r}; supported values: {ACTIVE_SYSTEMS}")
    spec = SYSTEM_SPECS[system_name]
    return {
        "name": spec.name,
        "label": spec.label,
        "atom": spec.atom,
        "basis": spec.basis,
        "charge": spec.charge,
        "spin": spec.spin,
        "unit": spec.unit,
    }


def runtime_config(default_config: RunConfig) -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, default_config, RunConfig)


def _afqmc_seed(system_name: str) -> int:
    digest = hashlib.sha256(f"afqmc-ipie::{system_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def live_objective_score(final_energy: float, block_energy_stderr: float) -> float:
    return float(final_energy + LIVE_OBJECTIVE_STDERR_WEIGHT * block_energy_stderr)


def _energy_column(frame) -> str:
    for candidate in ("ETotal", "TotalEnergy", "LocalEnergy", "Energy"):
        if candidate in frame.columns:
            return candidate
    raise RuntimeError(f"could not locate an energy column in ipie estimator output: {list(frame.columns)}")


def _weight_column(frame) -> str | None:
    for candidate in ("Weight", "WeightFactor"):
        if candidate in frame.columns:
            return candidate
    return None


def _production_start(total_observed_blocks: int) -> int:
    if total_observed_blocks <= 1:
        raise RuntimeError("AFQMC estimator output did not contain propagated blocks")
    measured_blocks = total_observed_blocks - 1
    tail_blocks = max(MIN_PRODUCTION_BLOCKS, int(np.ceil(SCORING_TAIL_FRACTION * measured_blocks)))
    tail_blocks = min(tail_blocks, measured_blocks)
    return 1 + max(0, measured_blocks - tail_blocks)


def _scratch_root() -> Path | None:
    value = os.environ.get("AUTORESEARCH_AFQMC_WORKDIR_ROOT")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_config(cfg: RunConfig, system_name: str, wall_time_limit: float, target_energy: float) -> dict:
    validate_runtime_config(cfg)
    if system_name not in ACTIVE_SYSTEMS:
        raise ValueError(f"unsupported molecular AFQMC system {system_name!r}; supported values: {ACTIVE_SYSTEMS}")
    AFQMC, extract_observable, gen_ipie_input_from_pyscf_chk = _import_ipie_modules()

    total_start = time.perf_counter()
    spec = SYSTEM_SPECS[system_name]
    mol = build_molecule(spec)
    build_wall = time.perf_counter() - total_start

    comm = _try_import_mpi_comm()
    rank = int(comm.Get_rank()) if comm is not None else 0
    mpi_size = int(comm.Get_size()) if comm is not None else 1
    scratch_root = _scratch_root()
    temp_ctx = None
    if mpi_size > 1:
        root_workdir = Path(tempfile.mkdtemp(prefix=f".afqmc_ipie_{system_name}_", dir=scratch_root)) if rank == 0 else None
        workdir = Path(comm.bcast(str(root_workdir) if root_workdir is not None else None, root=0))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f".afqmc_ipie_{system_name}_", dir=scratch_root)
        workdir = Path(temp_ctx.__enter__())
    try:
        chk_file = workdir / "scf.chk"
        hamil_file = workdir / "hamiltonian.h5"
        wfn_file = workdir / "wavefunction.h5"
        estimator_file = workdir / "estimates.0.h5"

        scf_energy = None
        scf_wall = 0.0
        input_wall = 0.0
        cycle_energies: list[float] = []
        scf_converged = False
        scf_cycles = 0
        if rank == 0:
            solver = build_solver(mol, cfg)
            solver.chkfile = str(chk_file)
            dm0 = solver.get_init_guess(key=cfg.init_guess)
            previous_callback = getattr(solver, "callback", None)

            def _capture_cycle(envs):
                energy = envs.get("e_tot")
                if energy is not None:
                    cycle_energies.append(float(energy))
                if previous_callback is not None:
                    previous_callback(envs)

            solver.callback = _capture_cycle
            scf_start = time.perf_counter()
            scf_energy = float(solver.kernel(dm0=dm0))
            scf_wall = time.perf_counter() - scf_start
            scf_converged = bool(getattr(solver, "converged", False))
            scf_cycles = int(getattr(solver, "cycles", 0) or 0)
            if not scf_converged:
                raise RuntimeError("SCF trial-state build did not converge")
            if not cycle_energies or abs(cycle_energies[-1] - scf_energy) > 1e-12:
                cycle_energies.append(scf_energy)

            input_start = time.perf_counter()
            gen_ipie_input_from_pyscf_chk(
                str(chk_file),
                hamil_file=str(hamil_file),
                wfn_file=str(wfn_file),
                verbose=False,
                chol_cut=cfg.chol_cut,
            )
            input_wall = time.perf_counter() - input_start
        if comm is not None:
            comm.Barrier()

        afqmc_start = time.perf_counter()
        driver = AFQMC.build_from_hdf5(
            mol.nelec,
            str(hamil_file),
            str(wfn_file),
            num_walkers=cfg.num_walkers_per_rank,
            seed=_afqmc_seed(system_name),
            num_steps_per_block=cfg.num_steps_per_block,
            num_blocks=cfg.num_blocks,
            timestep=cfg.timestep,
            stabilize_freq=cfg.stabilize_freq,
            pop_control_freq=cfg.pop_control_freq,
            verbose=False,
        )
        driver.run(estimator_filename=str(estimator_file), verbose=False)
        driver.finalise(verbose=False)
        afqmc_wall = time.perf_counter() - afqmc_start
        if comm is not None:
            comm.Barrier()
        if rank != 0:
            return None

        if not estimator_file.exists():
            raise RuntimeError("ipie did not produce an estimator file")
        energy_frame = extract_observable(str(estimator_file), "energy")
        if len(energy_frame) <= 1:
            raise RuntimeError("ipie estimator output did not contain enough blocks")

        energy_key = _energy_column(energy_frame)
        weight_key = _weight_column(energy_frame)
        block_ids = (
            np.asarray(energy_frame["Block"], dtype=float)
            if "Block" in energy_frame.columns
            else np.arange(len(energy_frame), dtype=float)
        )
        mixed_estimator_energies = np.asarray(energy_frame[energy_key], dtype=float)
        tau_grid = block_ids * float(cfg.num_steps_per_block) * float(cfg.timestep)
        production_start = _production_start(len(mixed_estimator_energies))
        if len(mixed_estimator_energies) - production_start < MIN_PRODUCTION_BLOCKS:
            raise RuntimeError(
                f"AFQMC run produced only {len(mixed_estimator_energies) - production_start} production blocks; "
                f"need at least {MIN_PRODUCTION_BLOCKS}"
            )

        production_energies = mixed_estimator_energies[production_start:]
        production_tau = tau_grid[production_start:]
        block_energy_std = float(np.std(production_energies, ddof=1))
        block_energy_stderr = block_energy_std / float(np.sqrt(len(production_energies)))
        final_energy = float(np.mean(production_energies))
        score = live_objective_score(final_energy, block_energy_stderr)
        final_error = final_energy - target_energy
        abs_final_error = abs(final_error)
        total_wall = time.perf_counter() - total_start

        weight_trace = None
        if weight_key is not None:
            weight_trace = [float(value) for value in np.asarray(energy_frame[weight_key], dtype=float)]

        return {
            "config": config_to_dict(cfg),
            "system": system_name,
            "label": spec.label,
            "atom": spec.atom,
            "basis": spec.basis,
            "charge": spec.charge,
            "spin": spec.spin,
            "unit": spec.unit,
            "metric": LIVE_OBJECTIVE_METRIC,
            "lower_is_better": True,
            "score": score,
            "risk_adjusted_energy": score,
            "objective_stderr_weight": LIVE_OBJECTIVE_STDERR_WEIGHT,
            "scoring_tail_fraction": SCORING_TAIL_FRACTION,
            "reference_energy": target_energy,
            "sampling_method": "ipie_phaseless_afqmc_blocks",
            "seed": _afqmc_seed(system_name),
            "tau_grid": [float(value) for value in tau_grid],
            "production_tau_grid": [float(value) for value in production_tau],
            "mixed_estimator_energies": [float(value) for value in mixed_estimator_energies],
            "block_averaged_energies": [float(value) for value in production_energies],
            "equilibration_block_count": int(production_start),
            "discarded_block_count": int(production_start),
            "tail_start_tau": float(production_tau[0]),
            "block_count_total": int(len(mixed_estimator_energies)),
            "block_count": int(len(production_energies)),
            "block_energy_std": block_energy_std,
            "block_energy_stderr": block_energy_stderr,
            "tail_energy_std": block_energy_std,
            "tail_energy_stderr": block_energy_stderr,
            "walker_population": weight_trace,
            "trial_energy": scf_energy,
            "scf_energy": scf_energy,
            "final_energy": final_energy,
            "final_error": final_error,
            "abs_final_error": abs_final_error,
            "scf_converged": scf_converged,
            "scf_cycles": scf_cycles,
            "cycle_energies": cycle_energies,
            "num_walkers_per_rank": int(cfg.num_walkers_per_rank),
            "num_walkers_total": int(cfg.num_walkers_per_rank * driver.mpi_handler.size),
            "mpi_size": mpi_size,
            "num_steps_per_block": int(cfg.num_steps_per_block),
            "num_blocks_requested": int(cfg.num_blocks),
            "timestep": float(cfg.timestep),
            "stabilize_freq": int(cfg.stabilize_freq),
            "pop_control_freq": int(cfg.pop_control_freq),
            "wall_seconds": total_wall,
            "build_wall_seconds": build_wall,
            "scf_wall_seconds": scf_wall,
            "input_wall_seconds": input_wall,
            "afqmc_wall_seconds": afqmc_wall,
            "total_wall_seconds": total_wall,
            "wall_budget_seconds": wall_time_limit,
            "supported_systems": list(ACTIVE_SYSTEMS),
        }
    finally:
        if comm is not None:
            comm.Barrier()
        if mpi_size > 1:
            if rank == 0:
                shutil.rmtree(workdir, ignore_errors=True)
        elif temp_ctx is not None:
            temp_ctx.__exit__(None, None, None)


def evaluate_source_file(source_file: Path, wall_time_limit: float) -> dict:
    system_name, cfg = load_source_definition(source_file)
    target = reference_energy(system_name)
    if target is None:
        raise RuntimeError(f"missing reference energy for {system_name}; run compute_reference_energies.py first")
    return run_config(cfg, system_name, wall_time_limit=wall_time_limit, target_energy=target)


def compute_reference_record(system_name: str) -> dict:
    if system_name not in ACTIVE_SYSTEMS:
        raise ValueError(f"unsupported molecular AFQMC system {system_name!r}; supported values: {ACTIVE_SYSTEMS}")
    spec = SYSTEM_SPECS[system_name]
    start = time.perf_counter()
    mol = build_molecule(spec)
    build_wall = time.perf_counter() - start

    solver = scf.RHF(mol)
    solver.conv_tol = REFERENCE_SCF_CONFIG.scf_conv_tol
    solver.max_cycle = REFERENCE_SCF_CONFIG.scf_max_cycle
    solver.diis_space = REFERENCE_SCF_CONFIG.diis_space
    solver.level_shift = REFERENCE_SCF_CONFIG.level_shift
    solver.damp = REFERENCE_SCF_CONFIG.damping
    solver.verbose = 0

    dm0 = solver.get_init_guess(key=REFERENCE_SCF_CONFIG.init_guess)
    scf_start = time.perf_counter()
    hf_energy = float(solver.kernel(dm0=dm0))
    scf_wall = time.perf_counter() - scf_start
    if not bool(getattr(solver, "converged", False)):
        raise RuntimeError(f"reference SCF did not converge for {system_name}")

    ccsd_solver = cc.CCSD(solver)
    ccsd_solver.conv_tol = REFERENCE_CCSD_CONV_TOL
    ccsd_solver.conv_tol_normt = REFERENCE_CCSD_CONV_TOL_NORMT
    ccsd_solver.max_cycle = REFERENCE_CCSD_MAX_CYCLE
    ccsd_start = time.perf_counter()
    corr_energy, *_ = ccsd_solver.kernel()
    ccsd_wall = time.perf_counter() - ccsd_start
    if not bool(getattr(ccsd_solver, "converged", False)):
        raise RuntimeError(f"reference CCSD did not converge for {system_name}")

    triples_start = time.perf_counter()
    triples_energy = float(ccsd_solver.ccsd_t())
    triples_wall = time.perf_counter() - triples_start
    reference_total = hf_energy + float(corr_energy) + triples_energy

    ccsdt_reference = {
        "method_key": "ccsd_t",
        "reference_method": "CCSD(T)",
        "convergence_mode": "offline_unconstrained_reference",
        "reference_energy": reference_total,
        "scf_energy": hf_energy,
        "correlation_energy": float(corr_energy),
        "triples_correction": float(triples_energy),
        "converged": bool(getattr(solver, "converged", False)),
        "reference_converged": bool(getattr(ccsd_solver, "converged", False)),
        "wall_seconds": time.perf_counter() - start,
        "build_wall_seconds": build_wall,
        "scf_wall_seconds": scf_wall,
        "ccsd_wall_seconds": ccsd_wall,
        "triples_wall_seconds": triples_wall,
        "scf_config": {
            "trial": REFERENCE_SCF_CONFIG.trial,
            "scf_conv_tol": REFERENCE_SCF_CONFIG.scf_conv_tol,
            "scf_max_cycle": REFERENCE_SCF_CONFIG.scf_max_cycle,
            "diis_space": REFERENCE_SCF_CONFIG.diis_space,
            "level_shift": REFERENCE_SCF_CONFIG.level_shift,
            "damping": REFERENCE_SCF_CONFIG.damping,
            "init_guess": REFERENCE_SCF_CONFIG.init_guess,
        },
        "ccsd_conv_tol": REFERENCE_CCSD_CONV_TOL,
        "ccsd_conv_tol_normt": REFERENCE_CCSD_CONV_TOL_NORMT,
        "ccsd_max_cycle": REFERENCE_CCSD_MAX_CYCLE,
    }
    return {
        "trial": REFERENCE_SCF_CONFIG.trial,
        "primary_reference": "ccsd_t",
        "reference_method": ccsdt_reference["reference_method"],
        "reference_energy": ccsdt_reference["reference_energy"],
        "converged": ccsdt_reference["converged"],
        "reference_converged": ccsdt_reference["reference_converged"],
        "wall_seconds": ccsdt_reference["wall_seconds"],
        "build_wall_seconds": ccsdt_reference["build_wall_seconds"],
        "scf_wall_seconds": ccsdt_reference["scf_wall_seconds"],
        "ccsd_wall_seconds": ccsdt_reference["ccsd_wall_seconds"],
        "triples_wall_seconds": ccsdt_reference["triples_wall_seconds"],
        "basis": spec.basis,
        "charge": spec.charge,
        "spin": spec.spin,
        "unit": spec.unit,
        "label": spec.label,
        "references": {
            "ccsd_t": ccsdt_reference,
        },
    }


def run_cli(system_name: str, default_config: RunConfig) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed molecular AFQMC benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    args = parser.parse_args()

    target = reference_energy(system_name)
    if target is None:
        raise SystemExit(f"missing reference energy for {system_name}; run compute_reference_energies.py first")
    result = run_config(runtime_config(default_config), system_name, wall_time_limit=args.wall_seconds, target_energy=target)
    print(json.dumps(result, indent=2))
    return 0
