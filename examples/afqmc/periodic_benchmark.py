"""
Periodic correlated-energy benchmark for the AFQMC lane surface.

The path remains `examples/afqmc/`, but the active examples are periodic
PySCF-PBC electronic systems scored by periodic MP2 total energy under a fixed
wall-time budget. Each evaluation also emits bootstrap block-averaged energy
estimates derived from the MP2 correlation contributions so downstream plots
have a real sample distribution to compare.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyscf.pbc import cc, gto, scf
from pyscf.pbc import mp as pbc_mp

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
BOOTSTRAP_BLOCK_COUNT = 64
MIN_TERMS_PER_BLOCK = 16
MAX_TERMS_PER_BLOCK = 64
ALLOWED_TRIALS = {"rhf", "uhf"}
ALLOWED_INIT_GUESSES = {"minao", "atom", "1e", "huckel", "mod_huckel"}
MAX_RUNTIME_CELL_PRECISION = 1e-3
MIN_RUNTIME_CELL_PRECISION = 1e-12
MAX_RUNTIME_CONV_TOL = 1e-2
MIN_RUNTIME_CONV_TOL = 1e-12
MAX_RUNTIME_MAX_CYCLE = 256
MAX_RUNTIME_DIIS_SPACE = 20
MAX_RUNTIME_LEVEL_SHIFT = 1.0
MAX_RUNTIME_DAMPING = 0.5
MAX_T2_ABS = 10.0
MAX_CORRELATION_ENERGY_PER_ELECTRON = 1.0


@dataclass(frozen=True)
class RunConfig:
    name: str
    trial: str
    cell_precision: float
    conv_tol: float
    max_cycle: int
    diis_space: int
    level_shift: float
    damping: float
    init_guess: str


REFERENCE_CONFIG = RunConfig(
    name="reference_periodic_scf",
    trial="rhf",
    cell_precision=1e-9,
    conv_tol=1e-10,
    max_cycle=96,
    diis_space=12,
    level_shift=0.0,
    damping=0.0,
    init_guess="minao",
)

REFERENCE_SCF_CONFIG = RunConfig(
    name="offline_reference_krhf",
    trial="rhf",
    cell_precision=1e-10,
    conv_tol=1e-12,
    max_cycle=256,
    diis_space=12,
    level_shift=0.0,
    damping=0.0,
    init_guess="minao",
)

REFERENCE_CCSD_CONV_TOL = 1e-9
REFERENCE_CCSD_CONV_TOL_NORMT = 1e-7
REFERENCE_CCSD_MAX_CYCLE = 256


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
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "RunConfig":
            if node.args:
                raise ValueError("RunConfig(...) must use keyword arguments only")
            payload = {keyword.arg: _safe_eval_node(keyword.value) for keyword in node.keywords if keyword.arg}
            return RunConfig(**payload)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "float"
            and node.func.attr == "fromhex"
        ):
            if len(node.args) != 1 or node.keywords:
                raise ValueError("float.fromhex(...) must take exactly one positional argument")
            arg = _safe_eval_node(node.args[0])
            if not isinstance(arg, str):
                raise ValueError("float.fromhex(...) expects a string literal")
            return float.fromhex(arg)
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
    if not (MIN_RUNTIME_CELL_PRECISION <= cfg.cell_precision <= MAX_RUNTIME_CELL_PRECISION):
        raise ValueError(f"cell_precision must be between {MIN_RUNTIME_CELL_PRECISION:g} and {MAX_RUNTIME_CELL_PRECISION:g}")
    if not (MIN_RUNTIME_CONV_TOL <= cfg.conv_tol <= MAX_RUNTIME_CONV_TOL):
        raise ValueError(f"conv_tol must be between {MIN_RUNTIME_CONV_TOL:g} and {MAX_RUNTIME_CONV_TOL:g}")
    if not (1 <= int(cfg.max_cycle) <= MAX_RUNTIME_MAX_CYCLE):
        raise ValueError(f"max_cycle must be between 1 and {MAX_RUNTIME_MAX_CYCLE}")
    if not (1 <= int(cfg.diis_space) <= MAX_RUNTIME_DIIS_SPACE):
        raise ValueError(f"diis_space must be between 1 and {MAX_RUNTIME_DIIS_SPACE}")
    if not (0.0 <= float(cfg.level_shift) <= MAX_RUNTIME_LEVEL_SHIFT):
        raise ValueError(f"level_shift must be between 0 and {MAX_RUNTIME_LEVEL_SHIFT}")
    if not (0.0 <= float(cfg.damping) <= MAX_RUNTIME_DAMPING):
        raise ValueError(f"damping must be between 0 and {MAX_RUNTIME_DAMPING}")
    if cfg.init_guess not in ALLOWED_INIT_GUESSES:
        raise ValueError(f"unsupported init_guess {cfg.init_guess!r}")


def build_cell(spec, cfg: RunConfig):
    cell = gto.Cell()
    cell.atom = spec.atom
    cell.a = [list(row) for row in spec.lattice_vectors]
    cell.unit = spec.unit
    cell.basis = spec.basis
    cell.pseudo = spec.pseudo
    cell.charge = spec.charge
    cell.spin = spec.spin
    cell.precision = cfg.cell_precision
    cell.verbose = 0
    cell.build()
    return cell


def build_solver(cell, cfg: RunConfig):
    if cfg.trial == "rhf":
        solver = scf.RHF(cell)
    elif cfg.trial == "uhf":
        solver = scf.UHF(cell)
    else:
        raise ValueError(f"unsupported trial {cfg.trial!r}")
    solver.conv_tol = cfg.conv_tol
    solver.max_cycle = cfg.max_cycle
    solver.diis_space = cfg.diis_space
    solver.level_shift = cfg.level_shift
    solver.damp = cfg.damping
    solver.verbose = 0
    return solver


def build_problem(system_name: str) -> dict:
    if system_name not in ACTIVE_SYSTEMS:
        raise ValueError(f"unsupported periodic system {system_name!r}; supported values: {ACTIVE_SYSTEMS}")
    spec = SYSTEM_SPECS[system_name]
    return {
        "name": spec.name,
        "label": spec.label,
        "atom": spec.atom,
        "lattice_vectors": [list(row) for row in spec.lattice_vectors],
        "basis": spec.basis,
        "pseudo": spec.pseudo,
        "charge": spec.charge,
        "spin": spec.spin,
        "unit": spec.unit,
    }


def runtime_config(default_config: RunConfig) -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, default_config, RunConfig)


def _bootstrap_seed(system_name: str) -> int:
    digest = hashlib.sha256(f"afqmc-mp2-bootstrap::{system_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _rhf_mp2_correlation_terms(t2, eris) -> np.ndarray:
    if not hasattr(eris, "ovov"):
        raise ValueError("periodic MP2 eris object does not expose ovov integrals")
    nocc, nvir = t2.shape[1:3]
    ovov = np.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    terms = (2.0 * t2 * ovov - t2 * ovov.swapaxes(2, 3)).real
    return terms.reshape(-1)


def _uhf_mp2_correlation_terms(t2, eris) -> np.ndarray:
    t2aa, t2ab, t2bb = (np.asarray(block) for block in t2)
    terms: list[np.ndarray] = []

    if t2aa.size:
        nocca, _, nvira, _ = t2aa.shape
        ovov = np.asarray(eris.ovov).reshape(nocca, nvira, nocca, nvira).transpose(0, 2, 1, 3)
        aa_terms = (0.5 * t2aa * ovov - 0.5 * t2aa * ovov.swapaxes(2, 3)).real
        terms.append(aa_terms.reshape(-1))

    if t2ab.size:
        nocca, noccb, nvira, nvirb = t2ab.shape
        ovOV = np.asarray(eris.ovOV).reshape(nocca, nvira, noccb, nvirb).transpose(0, 2, 1, 3)
        ab_terms = (t2ab * ovOV).real
        terms.append(ab_terms.reshape(-1))

    if t2bb.size:
        noccb, _, nvirb, _ = t2bb.shape
        OVOV = np.asarray(eris.OVOV).reshape(noccb, nvirb, noccb, nvirb).transpose(0, 2, 1, 3)
        bb_terms = (0.5 * t2bb * OVOV - 0.5 * t2bb * OVOV.swapaxes(2, 3)).real
        terms.append(bb_terms.reshape(-1))

    if not terms:
        return np.zeros(1, dtype=float)
    return np.concatenate(terms)


def _mp2_correlation_terms(mp2_solver, t2, eris) -> np.ndarray:
    if isinstance(t2, tuple):
        return _uhf_mp2_correlation_terms(t2, eris)
    return _rhf_mp2_correlation_terms(np.asarray(t2), eris)


def _bootstrap_block_averages(system_name: str, scf_energy: float, correlation_terms: np.ndarray) -> tuple[list[float], int]:
    term_count = int(correlation_terms.size)
    if term_count <= 0:
        return [float(scf_energy)], 1
    terms_per_block = min(MAX_TERMS_PER_BLOCK, max(MIN_TERMS_PER_BLOCK, int(np.ceil(np.sqrt(term_count)))))
    rng = np.random.default_rng(_bootstrap_seed(system_name))
    sample_index = rng.integers(0, term_count, size=(BOOTSTRAP_BLOCK_COUNT, terms_per_block))
    sampled_terms = correlation_terms[sample_index]
    block_corr = sampled_terms.mean(axis=1) * term_count
    return [float(scf_energy + value) for value in block_corr], terms_per_block


def run_config(cfg: RunConfig, system_name: str, wall_time_limit: float, target_energy: float) -> dict:
    validate_runtime_config(cfg)
    total_start = time.perf_counter()
    spec = SYSTEM_SPECS[system_name]
    cell = build_cell(spec, cfg)
    build_wall = time.perf_counter() - total_start

    solver = build_solver(cell, cfg)
    dm0 = solver.get_init_guess(key=cfg.init_guess)
    cycle_energies: list[float] = []
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
    if not bool(getattr(solver, "converged", False)):
        raise RuntimeError("SCF did not converge")
    if not cycle_energies or abs(cycle_energies[-1] - scf_energy) > 1e-14:
        cycle_energies.append(scf_energy)
    mp2_start = time.perf_counter()
    mp2_solver = pbc_mp.UMP2(solver) if cfg.trial == "uhf" else pbc_mp.MP2(solver)
    eris = mp2_solver.ao2mo()
    correlation_energy, t2 = mp2_solver.init_amps(eris=eris, with_t2=True)
    correlation_energy = float(correlation_energy)
    mp2_wall = time.perf_counter() - mp2_start
    if not np.isfinite(correlation_energy):
        raise RuntimeError("MP2 correlation energy was not finite")
    t2_max_abs = float(np.max(np.abs(t2))) if np.size(t2) else 0.0
    if not np.isfinite(t2_max_abs):
        raise RuntimeError("MP2 amplitudes were not finite")
    if t2_max_abs > MAX_T2_ABS:
        raise RuntimeError(f"MP2 amplitudes became numerically unstable (max |t2| = {t2_max_abs:.3e})")
    correlation_energy_limit = MAX_CORRELATION_ENERGY_PER_ELECTRON * max(int(cell.nelectron), 1)
    if abs(correlation_energy) > correlation_energy_limit:
        raise RuntimeError(
            f"MP2 correlation energy magnitude exceeded the stability limit ({correlation_energy:.6f} Ha)"
        )
    final_energy = scf_energy + correlation_energy
    correlation_terms = _mp2_correlation_terms(mp2_solver, t2, eris)
    if not np.all(np.isfinite(correlation_terms)):
        raise RuntimeError("MP2 correlation terms were not finite")
    block_averaged_energies, terms_per_block = _bootstrap_block_averages(system_name, scf_energy, correlation_terms)
    if not np.all(np.isfinite(np.asarray(block_averaged_energies, dtype=float))):
        raise RuntimeError("block-averaged energies were not finite")
    block_energy_std = float(np.std(block_averaged_energies, ddof=1)) if len(block_averaged_energies) > 1 else 0.0
    block_energy_stderr = block_energy_std / float(np.sqrt(len(block_averaged_energies))) if block_averaged_energies else 0.0
    final_error = final_energy - target_energy
    abs_final_error = abs(final_error)
    total_wall = time.perf_counter() - total_start
    return {
        "config": config_to_dict(cfg),
        "system": system_name,
        "label": spec.label,
        "basis": spec.basis,
        "pseudo": spec.pseudo,
        "metric": "final_energy",
        "lower_is_better": True,
        "score": final_energy,
        "sampling_method": "bootstrap_mp2_correlation_terms",
        "bootstrap_seed": _bootstrap_seed(system_name),
        "block_averaged_energies": block_averaged_energies,
        "block_count": len(block_averaged_energies),
        "terms_per_block": terms_per_block,
        "correlation_term_count": int(correlation_terms.size),
        "block_energy_std": block_energy_std,
        "block_energy_stderr": block_energy_stderr,
        "t2_max_abs": t2_max_abs,
        "reference_energy": target_energy,
        "scf_energy": scf_energy,
        "correlation_energy": correlation_energy,
        "final_energy": final_energy,
        "final_error": final_error,
        "abs_final_error": abs_final_error,
        "converged": bool(getattr(solver, "converged", False)),
        "scf_cycles": int(getattr(solver, "cycles", 0) or 0),
        "cycle_energies": cycle_energies,
        "nao": int(cell.nao_nr()),
        "nelectron": int(cell.nelectron),
        "wall_seconds": total_wall,
        "build_wall_seconds": build_wall,
        "scf_wall_seconds": scf_wall,
        "correlation_wall_seconds": mp2_wall,
        "total_wall_seconds": total_wall,
        "wall_budget_seconds": wall_time_limit,
        "supported_systems": list(ACTIVE_SYSTEMS),
    }


def evaluate_source_file(source_file: Path, wall_time_limit: float) -> dict:
    system_name, cfg = load_source_definition(source_file)
    if system_name not in ACTIVE_SYSTEMS:
        raise ValueError(f"unsupported periodic system {system_name!r}; supported values: {ACTIVE_SYSTEMS}")
    target = reference_energy(system_name)
    if target is None:
        raise RuntimeError(f"missing reference energy for {system_name}; run compute_reference_energies.py first")
    return run_config(cfg, system_name, wall_time_limit=wall_time_limit, target_energy=target)


def compute_reference_record(system_name: str) -> dict:
    spec = SYSTEM_SPECS[system_name]
    start = time.perf_counter()
    cfg = RunConfig(
        name="reference_krhf_krccsd_t",
        trial="rhf",
        cell_precision=REFERENCE_SCF_CONFIG.cell_precision,
        conv_tol=REFERENCE_SCF_CONFIG.conv_tol,
        max_cycle=REFERENCE_SCF_CONFIG.max_cycle,
        diis_space=REFERENCE_SCF_CONFIG.diis_space,
        level_shift=REFERENCE_SCF_CONFIG.level_shift,
        damping=REFERENCE_SCF_CONFIG.damping,
        init_guess=REFERENCE_SCF_CONFIG.init_guess,
    )
    build_start = time.perf_counter()
    cell = build_cell(spec, cfg)
    build_wall = time.perf_counter() - build_start
    kmesh = (1, 1, 1)
    kpts = cell.make_kpts(kmesh)
    solver = scf.KRHF(cell, kpts)
    solver.conv_tol = cfg.conv_tol
    solver.max_cycle = cfg.max_cycle
    solver.diis_space = cfg.diis_space
    solver.level_shift = cfg.level_shift
    solver.damp = cfg.damping
    solver.verbose = 0
    dm0 = solver.get_init_guess(key=cfg.init_guess)
    scf_start = time.perf_counter()
    hf_energy = float(solver.kernel(dm0=dm0))
    scf_wall = time.perf_counter() - scf_start
    ccsd_solver = cc.KRCCSD(solver)
    ccsd_solver.conv_tol = REFERENCE_CCSD_CONV_TOL
    ccsd_solver.conv_tol_normt = REFERENCE_CCSD_CONV_TOL_NORMT
    ccsd_solver.max_cycle = REFERENCE_CCSD_MAX_CYCLE
    ccsd_start = time.perf_counter()
    corr_energy, *_ = ccsd_solver.kernel()
    ccsd_wall = time.perf_counter() - ccsd_start
    eris_start = time.perf_counter()
    eris = ccsd_solver.ao2mo()
    eris_wall = time.perf_counter() - eris_start
    triples_start = time.perf_counter()
    triples_energy = complex(np.asarray(ccsd_solver.ccsd_t(eris=eris)).reshape(()))
    triples_wall = time.perf_counter() - triples_start
    triples_real = float(triples_energy.real)
    triples_imag = float(triples_energy.imag)
    reference_total = hf_energy + float(corr_energy) + triples_real
    ccsdt_reference = {
        "method_key": "ccsd_t",
        "reference_method": "CCSD(T)",
        "convergence_mode": "offline_unconstrained_reference",
        "scf_energy": hf_energy,
        "correlation_energy": float(corr_energy),
        "triples_correction": triples_real,
        "triples_correction_imag": triples_imag,
        "reference_energy": reference_total,
        "converged": bool(getattr(solver, "converged", False)),
        "reference_converged": bool(getattr(ccsd_solver, "converged", False)),
        "wall_seconds": time.perf_counter() - start,
        "build_wall_seconds": build_wall,
        "scf_wall_seconds": scf_wall,
        "ccsd_wall_seconds": ccsd_wall,
        "eris_wall_seconds": eris_wall,
        "triples_wall_seconds": triples_wall,
        "kmesh": list(kmesh),
        "scf_config": config_to_dict(cfg),
        "ccsd_conv_tol": REFERENCE_CCSD_CONV_TOL,
        "ccsd_conv_tol_normt": REFERENCE_CCSD_CONV_TOL_NORMT,
        "ccsd_max_cycle": REFERENCE_CCSD_MAX_CYCLE,
    }
    return {
        "trial": cfg.trial,
        "primary_reference": "ccsd_t",
        "reference_method": ccsdt_reference["reference_method"],
        "reference_energy": ccsdt_reference["reference_energy"],
        "converged": ccsdt_reference["converged"],
        "reference_converged": ccsdt_reference["reference_converged"],
        "wall_seconds": ccsdt_reference["wall_seconds"],
        "build_wall_seconds": ccsdt_reference["build_wall_seconds"],
        "scf_wall_seconds": ccsdt_reference["scf_wall_seconds"],
        "ccsd_wall_seconds": ccsdt_reference["ccsd_wall_seconds"],
        "eris_wall_seconds": ccsdt_reference["eris_wall_seconds"],
        "triples_wall_seconds": ccsdt_reference["triples_wall_seconds"],
        "basis": spec.basis,
        "pseudo": spec.pseudo,
        "nao": int(cell.nao_nr()),
        "nelectron": int(cell.nelectron),
        "label": spec.label,
        "kmesh": list(kmesh),
        "references": {
            "ccsd_t": ccsdt_reference,
        },
    }


def run_cli(system_name: str, default_config: RunConfig) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed periodic electronic benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    args = parser.parse_args()

    target = reference_energy(system_name)
    if target is None:
        raise SystemExit(f"missing reference energy for {system_name}; run compute_reference_energies.py first")

    result = run_config(runtime_config(default_config), system_name, wall_time_limit=args.wall_seconds, target_energy=target)
    print(json.dumps(result, indent=2))
    return 0
