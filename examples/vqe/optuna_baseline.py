from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchkit.optuna_utils import (
    THREAD_BUDGET_ENV,
    acquire_slot,
    load_python_module,
    prepare_frozen_source,
    resolve_repo_path,
    slugify,
    study_root,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
QUEUE_ROOT = SCRIPT_DIR / "eval_queue"
TRACKED_SNAPSHOT_ROOT = SCRIPT_DIR / "optuna_tracked_snapshots"
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_VQE_CONFIG_JSON"
SUMMARY_HEADER = (
    "trial\tstatus\tobjective\tfinal_energy\ttarget_energy\tfinal_error\tchem_acc_step\t"
    "wall_seconds\tqueue_wait_seconds\ttrial_dir\n"
)
CRASH_PENALTY = 1e9
MAX_STEPS_CHOICES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def default_script_for(molecule: str) -> Path:
    return SCRIPT_DIR / molecule.lower().replace("+", "_plus") / "initial_script.py"


def append_summary(path: Path, row: dict):
    if not path.exists():
        path.write_text(SUMMARY_HEADER)
    chem_acc_step = "" if row["chem_acc_step"] is None else row["chem_acc_step"]
    with path.open("a") as handle:
        handle.write(
            f"{row['trial']}\t{row['status']}\t{row['objective']:.12f}\t{row['final_energy']:.12f}\t"
            f"{row['target_energy']:.12f}\t{row['final_error']:.12e}\t{chem_acc_step}\t"
            f"{row['wall_seconds']:.4f}\t{row['queue_wait_seconds']:.4f}\t{row['trial_dir']}\n"
        )


def build_sampler(kind: str, seed: int, startup_trials: int):
    if kind == "gp":
        return optuna.samplers.GPSampler(
            seed=seed,
            n_startup_trials=startup_trials,
            deterministic_objective=True,
            independent_sampler=optuna.samplers.RandomSampler(seed=seed + 1),
            warn_independent_sampling=True,
        )
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)
    raise ValueError(f"unsupported sampler {kind!r}")


def choose_init_scale(trial: optuna.Trial, name: str) -> float:
    choices = [0.0, 1e-6, 1e-5, 1e-4, 6.25e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 1.8e-1, 2e-1, 3e-1, 4e-1, 5e-1]
    return float(trial.suggest_categorical(name, choices))


def ensure_nonzero_init_scale(value: float) -> float:
    return float(value if value > 0.0 else 1e-6)


def choose_max_steps(trial: optuna.Trial, name: str, minimum: int = 64) -> int:
    choices = [value for value in MAX_STEPS_CHOICES if value >= minimum]
    return int(trial.suggest_categorical(name, choices))


def choose_seed(trial: optuna.Trial, name: str = "seed") -> int:
    return int(trial.suggest_int(name, 0, 65535))


def apply_start_mode(
    trial: optuna.Trial,
    config: dict,
    start_mode_name: str,
    init_scale_name: str,
    reference: tuple[float, ...] | list[float],
) -> None:
    mode = trial.suggest_categorical(
        start_mode_name,
        ["zero", "random", "reference", "reference_noisy", "neg_reference"],
    )
    reference_tuple = tuple(float(value) for value in reference)
    if mode == "zero" or not reference_tuple:
        config["initial_parameters"] = ()
        config["init_scale"] = choose_init_scale(trial, init_scale_name)
        return
    if mode == "random":
        config["initial_parameters"] = ()
        config["init_scale"] = ensure_nonzero_init_scale(choose_init_scale(trial, init_scale_name))
        return
    if mode == "reference":
        config["initial_parameters"] = reference_tuple
        config["init_scale"] = choose_init_scale(trial, init_scale_name)
        return
    if mode == "reference_noisy":
        config["initial_parameters"] = reference_tuple
        config["init_scale"] = ensure_nonzero_init_scale(choose_init_scale(trial, init_scale_name))
        return
    config["initial_parameters"] = tuple(-value for value in reference_tuple)
    config["init_scale"] = choose_init_scale(trial, init_scale_name)


def bh_reference_initial(module) -> tuple[float, ...]:
    return tuple(float(value) for value in getattr(module.DEFAULT_CONFIG, "initial_parameters", ()))


def beh2_reference_initial(module) -> tuple[float, ...]:
    return tuple(float(value) for value in getattr(module.DEFAULT_CONFIG, "initial_parameters", ()))


def lih_reference_initial(module, param_model: str) -> tuple[float, ...]:
    full = [float(value) for value in getattr(module.DEFAULT_CONFIG, "initial_parameters", ())]
    if not full:
        return ()
    spec = module.MOLECULE_SPECS[module.MOLECULE_NAME]
    n_qubits = 2 * spec.active_orbitals
    singles = spec.active_electrons * (n_qubits - spec.active_electrons) // 2
    n_virtual = (n_qubits - spec.active_electrons) // 2
    single_block = full[:singles]
    double_block = full[singles:]
    diagonal = [double_block[index * n_virtual + index] for index in range(n_virtual)]
    off_diagonal = [
        0.5 * (double_block[0 * n_virtual + 1] + double_block[1 * n_virtual + 0]),
        0.5 * (double_block[0 * n_virtual + 2] + double_block[2 * n_virtual + 0]),
        0.5 * (double_block[1 * n_virtual + 2] + double_block[2 * n_virtual + 1]),
    ]
    paired_singles = [
        0.5 * (single_block[index] + single_block[index + n_virtual]) for index in range(n_virtual)
    ]

    if param_model == "uccsd_full":
        return tuple(full)
    if param_model == "pair_doubles_diag":
        return tuple(diagonal)
    if param_model == "pair_doubles_symmetric":
        return tuple(diagonal + off_diagonal)
    if param_model == "pair_doubles_full":
        return tuple(double_block)
    if param_model == "spin_paired_symmetric":
        return tuple(paired_singles + diagonal + off_diagonal)
    if param_model == "spin_paired_full":
        return tuple(paired_singles + double_block)
    raise ValueError(f"unknown LiH parameter model {param_model!r}")


def h2o_reference_initial(module, param_model: str) -> tuple[float, ...]:
    full = [float(value) for value in module.REFERENCE_FULL_UCCSD]
    spec = module.MOLECULE_SPECS[module.MOLECULE_NAME]
    singles, doubles = module.tied_uccsd_counts(spec.active_electrons, 2 * spec.active_orbitals)
    single_block = full[:singles]
    double_block = full[singles : singles + doubles]
    double_triplet = [
        sum(double_block[index::3]) / len(double_block[index::3]) for index in range(3)
    ]

    if param_model == "uccsd_full":
        return tuple(full)
    if param_model == "uccsd_tied_sd":
        return (
            sum(single_block) / len(single_block),
            sum(double_block) / len(double_block),
        )
    if param_model == "uccsd_doubles_triplet":
        return tuple(double_triplet)
    if param_model == "uccsd_sd_triplet":
        return (sum(single_block) / len(single_block), *double_triplet)
    if param_model == "uccsd_spin_paired_symmetric":
        return tuple(float(value) for value in module.REFERENCE_SPIN_PAIRED_SYMMETRIC)
    if param_model == "uccsd_spin_paired_full_doubles":
        return tuple(float(value) for value in module.REFERENCE_SPIN_PAIRED_FULL_DOUBLES)
    raise ValueError(f"unknown H2O parameter model {param_model!r}")


def apply_beh2_schedule(trial: optuna.Trial, config: dict) -> None:
    stages = [part.strip() for part in str(config["optimizer"]).split("+") if part.strip()]
    if len(stages) <= 1:
        config["stage_fractions"] = ()
        config["stage_max_steps"] = ()
        return
    weights = [trial.suggest_float(f"stage_weight_{index}", 0.1, 1.0) for index in range(len(stages))]
    total = sum(weights)
    fractions = [weight / total for weight in weights]
    stage_steps: list[int] = []
    assigned = 0
    for index, fraction in enumerate(fractions):
        if index == len(fractions) - 1:
            step_count = max(1, config["max_steps"] - assigned)
        else:
            remaining = len(fractions) - index - 1
            step_count = max(1, int(round(config["max_steps"] * fraction)))
            max_allowed = config["max_steps"] - assigned - remaining
            step_count = min(step_count, max_allowed)
        stage_steps.append(step_count)
        assigned += step_count
    config["stage_fractions"] = tuple(fractions)
    config["stage_max_steps"] = tuple(stage_steps)


def initial_snapshot_config(molecule: str) -> tuple[dict | None, str | None]:
    stem = molecule.lower().replace("+", "_plus")
    result_path = SCRIPT_DIR / "snapshots" / stem / "iter_0001" / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        config = payload.get("best_config") or payload.get("config")
        if isinstance(config, dict):
            return config, str(result_path)
    return None, None


def initial_trial_params(molecule: str, module) -> tuple[dict, str]:
    config, source = initial_snapshot_config(molecule)
    if config is None:
        config = asdict(module.DEFAULT_CONFIG)
        source = "module_default"

    if molecule == "BH":
        params = {
            "ansatz": str(config["ansatz"]),
            "optimizer": str(config["optimizer"]),
            "max_steps": int(config["max_steps"]),
            "init_scale": float(config["init_scale"]),
            "seed": int(config["seed"]),
            "cobyla_rhobeg": float(config.get("cobyla_rhobeg", asdict(module.DEFAULT_CONFIG)["cobyla_rhobeg"])),
            "cobyla_tol": float(config.get("cobyla_tol", asdict(module.DEFAULT_CONFIG)["cobyla_tol"])),
            "powell_xtol": float(config.get("powell_xtol", asdict(module.DEFAULT_CONFIG)["powell_xtol"])),
            "powell_ftol": float(config.get("powell_ftol", asdict(module.DEFAULT_CONFIG)["powell_ftol"])),
            "nelder_mead_xatol": float(
                config.get("nelder_mead_xatol", asdict(module.DEFAULT_CONFIG)["nelder_mead_xatol"])
            ),
            "nelder_mead_fatol": float(
                config.get("nelder_mead_fatol", asdict(module.DEFAULT_CONFIG)["nelder_mead_fatol"])
            ),
        }
        if params["ansatz"] != "uccsd":
            params["layers"] = int(config["layers"])
        return params, source

    if molecule == "LiH":
        defaults = asdict(module.DEFAULT_CONFIG)
        params = {
            "param_model": str(config["param_model"]),
            "optimizer": str(config["optimizer"]),
            "max_steps": int(config["max_steps"]),
            "seed": int(config["seed"]),
            "init_scale": float(config["init_scale"]),
            "step_size": float(config.get("step_size", defaults["step_size"])),
            "min_step_size": float(config.get("min_step_size", defaults["min_step_size"])),
            "cobyla_rhobeg": float(config.get("cobyla_rhobeg", defaults["cobyla_rhobeg"])),
            "cobyla_tol": float(config.get("cobyla_tol", defaults["cobyla_tol"])),
            "powell_xtol": float(config.get("powell_xtol", defaults["powell_xtol"])),
            "powell_ftol": float(config.get("powell_ftol", defaults["powell_ftol"])),
        }
        return params, source

    if molecule == "BeH2":
        defaults = asdict(module.DEFAULT_CONFIG)
        params = {
            "optimizer": str(config["optimizer"]),
            "max_steps": int(config["max_steps"]),
            "init_scale": float(config["init_scale"]),
            "seed": int(config["seed"]),
            "cobyla_rhobeg": float(config.get("cobyla_rhobeg", defaults["cobyla_rhobeg"])),
            "cobyla_tol": float(config.get("cobyla_tol", defaults["cobyla_tol"])),
            "powell_xtol": float(config.get("powell_xtol", defaults["powell_xtol"])),
            "powell_ftol": float(config.get("powell_ftol", defaults["powell_ftol"])),
        }
        if "+" in params["optimizer"]:
            stage_fractions = tuple(config.get("stage_fractions", ()) or ())
            if stage_fractions:
                params["stage1_fraction"] = float(stage_fractions[0])
            else:
                params["stage1_fraction"] = 0.5
        return params, source

    if molecule == "H2O":
        defaults = asdict(module.DEFAULT_CONFIG)
        params = {
            "param_model": str(config["param_model"]),
            "optimizer": str(config["optimizer"]),
            "max_steps": int(config["max_steps"]),
            "init_scale": float(config["init_scale"]),
            "seed": int(config["seed"]),
            "step_size": float(config.get("step_size", defaults["step_size"])),
            "min_step_size": float(config.get("min_step_size", defaults["min_step_size"])),
        }
        return params, source

    if molecule == "N2":
        params = {
            "ansatz": str(config["ansatz"]),
            "layers": int(config["layers"]),
            "optimizer": str(config["optimizer"]),
            "max_steps": int(config["max_steps"]),
            "init_scale": float(config["init_scale"]),
            "seed": int(config["seed"]),
        }
        return params, source

    raise ValueError(f"unsupported VQE molecule {molecule!r}")


def suggest_bh_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    ansatz = trial.suggest_categorical("ansatz", ["hea_ry_ring", "hea_ryrz_ring", "uccsd"])
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["ansatz"] = ansatz
    config["layers"] = 0 if ansatz == "uccsd" else trial.suggest_int("layers", 1, 10)
    config["optimizer"] = trial.suggest_categorical("optimizer", ["cobyla", "powell", "nelder-mead"])
    config["max_steps"] = choose_max_steps(trial, "max_steps")
    config["seed"] = choose_seed(trial)
    if ansatz == "uccsd":
        apply_start_mode(trial, config, "start_mode", "init_scale", bh_reference_initial(module))
    else:
        config["initial_parameters"] = ()
        config["init_scale"] = ensure_nonzero_init_scale(choose_init_scale(trial, "init_scale"))
    config["cobyla_rhobeg"] = trial.suggest_float("cobyla_rhobeg", 1e-5, 1.0, log=True)
    config["cobyla_tol"] = trial.suggest_float("cobyla_tol", 1e-14, 1e-3, log=True)
    config["powell_xtol"] = trial.suggest_float("powell_xtol", 1e-12, 1e-1, log=True)
    config["powell_ftol"] = trial.suggest_float("powell_ftol", 1e-14, 1e-3, log=True)
    config["nelder_mead_xatol"] = trial.suggest_float("nelder_mead_xatol", 1e-8, 1e-1, log=True)
    config["nelder_mead_fatol"] = trial.suggest_float("nelder_mead_fatol", 1e-12, 1e-3, log=True)
    return config


def suggest_lih_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    param_model = trial.suggest_categorical(
        "param_model",
        [
            "uccsd_full",
            "pair_doubles_diag",
            "pair_doubles_symmetric",
            "pair_doubles_full",
            "spin_paired_symmetric",
            "spin_paired_full",
        ],
    )
    optimizer = trial.suggest_categorical("optimizer", ["coordinate", "cobyla", "powell", "nelder-mead"])
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["param_model"] = param_model
    config["optimizer"] = optimizer
    config["max_steps"] = choose_max_steps(trial, "max_steps")
    config["seed"] = choose_seed(trial)
    apply_start_mode(trial, config, "start_mode", "init_scale", lih_reference_initial(module, param_model))
    config["step_size"] = trial.suggest_float("step_size", 1e-6, 5e-1, log=True)
    config["min_step_size"] = min(
        config["step_size"],
        trial.suggest_float("min_step_size", 1e-9, 1e-2, log=True),
    )
    config["cobyla_rhobeg"] = trial.suggest_float("cobyla_rhobeg", 1e-5, 1.0, log=True)
    config["cobyla_tol"] = trial.suggest_float("cobyla_tol", 1e-14, 1e-3, log=True)
    config["powell_xtol"] = trial.suggest_float("powell_xtol", 1e-10, 1e-1, log=True)
    config["powell_ftol"] = trial.suggest_float("powell_ftol", 1e-14, 1e-3, log=True)
    config["nelder_mead_xatol"] = trial.suggest_float("nelder_mead_xatol", 1e-8, 1e-1, log=True)
    config["nelder_mead_fatol"] = trial.suggest_float("nelder_mead_fatol", 1e-12, 1e-3, log=True)
    return config


def suggest_beh2_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    optimizer = trial.suggest_categorical(
        "optimizer",
        [
            "cobyla",
            "powell",
            "cobyla+powell",
            "powell+cobyla",
            "cobyla+powell+cobyla",
            "powell+cobyla+powell",
            "cobyla+powell+cobyla+powell",
            "powell+cobyla+powell+cobyla",
        ],
    )
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["optimizer"] = optimizer
    config["max_steps"] = choose_max_steps(trial, "max_steps", minimum=128)
    config["seed"] = choose_seed(trial)
    apply_start_mode(trial, config, "start_mode", "init_scale", beh2_reference_initial(module))
    config["cobyla_rhobeg"] = trial.suggest_float("cobyla_rhobeg", 1e-5, 5e-1, log=True)
    config["cobyla_tol"] = trial.suggest_float("cobyla_tol", 1e-14, 1e-3, log=True)
    config["powell_xtol"] = trial.suggest_float("powell_xtol", 1e-10, 1e-1, log=True)
    config["powell_ftol"] = trial.suggest_float("powell_ftol", 1e-14, 1e-3, log=True)
    apply_beh2_schedule(trial, config)
    return config


def suggest_h2o_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["param_model"] = trial.suggest_categorical(
        "param_model",
        [
            "uccsd_full",
            "uccsd_tied_sd",
            "uccsd_doubles_triplet",
            "uccsd_sd_triplet",
            "uccsd_spin_paired_symmetric",
            "uccsd_spin_paired_full_doubles",
        ],
    )
    config["optimizer"] = trial.suggest_categorical("optimizer", ["coordinate", "cobyla", "powell", "nelder-mead"])
    config["max_steps"] = choose_max_steps(trial, "max_steps")
    config["seed"] = choose_seed(trial)
    apply_start_mode(trial, config, "start_mode", "init_scale", h2o_reference_initial(module, config["param_model"]))
    config["step_size"] = trial.suggest_float("step_size", 1e-5, 5e-1, log=True)
    config["min_step_size"] = min(
        config["step_size"],
        trial.suggest_float("min_step_size", 1e-6, 1e-2, log=True),
    )
    config["cobyla_rhobeg"] = trial.suggest_float("cobyla_rhobeg", 1e-5, 1.0, log=True)
    config["cobyla_tol"] = trial.suggest_float("cobyla_tol", 1e-14, 1e-3, log=True)
    config["powell_xtol"] = trial.suggest_float("powell_xtol", 1e-10, 1e-1, log=True)
    config["powell_ftol"] = trial.suggest_float("powell_ftol", 1e-14, 1e-3, log=True)
    config["nelder_mead_xatol"] = trial.suggest_float("nelder_mead_xatol", 1e-8, 1e-1, log=True)
    config["nelder_mead_fatol"] = trial.suggest_float("nelder_mead_fatol", 1e-12, 1e-3, log=True)
    return config


def suggest_n2_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["ansatz"] = trial.suggest_categorical("ansatz", ["hea_ry_ring", "hea_ryrz_ring"])
    config["layers"] = trial.suggest_int("layers", 1, 10)
    config["optimizer"] = trial.suggest_categorical("optimizer", ["cobyla", "powell", "nelder-mead"])
    config["max_steps"] = choose_max_steps(trial, "max_steps")
    config["initial_parameters"] = ()
    config["init_scale"] = ensure_nonzero_init_scale(choose_init_scale(trial, "init_scale"))
    config["seed"] = choose_seed(trial)
    config["cobyla_rhobeg"] = trial.suggest_float("cobyla_rhobeg", 1e-5, 1.0, log=True)
    config["cobyla_tol"] = trial.suggest_float("cobyla_tol", 1e-14, 1e-3, log=True)
    config["powell_xtol"] = trial.suggest_float("powell_xtol", 1e-10, 1e-1, log=True)
    config["powell_ftol"] = trial.suggest_float("powell_ftol", 1e-14, 1e-3, log=True)
    config["nelder_mead_xatol"] = trial.suggest_float("nelder_mead_xatol", 1e-8, 1e-1, log=True)
    config["nelder_mead_fatol"] = trial.suggest_float("nelder_mead_fatol", 1e-12, 1e-3, log=True)
    return config


def suggest_vqe_config(trial: optuna.Trial, module) -> dict:
    molecule = module.MOLECULE_NAME
    if molecule == "BH":
        return suggest_bh_config(trial, module)
    if molecule == "LiH":
        return suggest_lih_config(trial, module)
    if molecule == "BeH2":
        return suggest_beh2_config(trial, module)
    if molecule == "H2O":
        return suggest_h2o_config(trial, module)
    if molecule == "N2":
        return suggest_n2_config(trial, module)
    raise ValueError(f"unsupported VQE molecule {molecule!r}")


def write_best_result(path: Path, study: optuna.Study):
    complete_trials = [trial for trial in study.trials if trial.value is not None]
    if not complete_trials:
        return
    best = study.best_trial
    payload = {
        "number": best.number,
        "value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    write_json(path, payload)


def main():
    parser = argparse.ArgumentParser(description="Run the Optuna VQE baseline on a frozen molecule-specific script.")
    parser.add_argument("--script", help="Path to the molecule-specific initial_script.py file.")
    parser.add_argument("--molecule", required=True, choices=["BH", "LiH", "BeH2", "H2O", "N2"])
    parser.add_argument("--archive-root", help="Optional directory to store this Optuna run.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--trials", type=int, default=100, help="Target total number of archived Optuna trials.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--sampler", choices=["gp", "tpe", "random"], default="tpe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--reset", action="store_true", help="Delete any existing Optuna archive for this molecule first.")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    source_file = resolve_repo_path(REPO_ROOT, args.script) if args.script else default_script_for(args.molecule)
    label = args.molecule
    root = resolve_repo_path(REPO_ROOT, args.archive_root) if args.archive_root else study_root(SCRIPT_DIR, label)
    tracked_snapshot_root = root / "tracked_snapshots"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    frozen_source, live_hash, frozen_hash = prepare_frozen_source(
        lane_dir=SCRIPT_DIR,
        label=label,
        source_file=source_file,
        reset=args.reset,
        archive_root=root,
    )
    module = load_python_module(frozen_source, f"vqe_optuna_source_{slugify(label)}")

    manifest = {
        "lane": "vqe",
        "label": label,
        "molecule": module.MOLECULE_NAME,
        "live_source": str(source_file),
        "frozen_source": str(frozen_source),
        "live_source_sha256": live_hash,
        "frozen_source_sha256": frozen_hash,
        "wall_seconds": args.wall_seconds,
        "trials_target": args.trials,
        "sampler": args.sampler,
        "sampler_seed": args.seed,
        "startup_trials": args.startup_trials,
        "max_parallel": args.max_parallel,
        "tracked_snapshot_root": str(tracked_snapshot_root),
    }
    initial_full_config, initial_source = initial_snapshot_config(args.molecule)
    if initial_full_config is None:
        initial_full_config = asdict(module.DEFAULT_CONFIG)
        initial_source = "module_default"
    initial_params, _ = initial_trial_params(args.molecule, module)
    manifest["initial_trial_params"] = initial_params
    manifest["initial_trial_source"] = initial_source
    write_json(root / "manifest.json", manifest)

    storage = f"sqlite:///{(root / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=f"vqe_{slugify(label)}",
        direction="minimize",
        sampler=build_sampler(args.sampler, args.seed, args.startup_trials),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        if trial.number == 0:
            config = dict(initial_full_config)
        else:
            config = suggest_vqe_config(trial, module)
        trial_dir = root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        write_json(trial_dir / "config.json", config)

        description = f"Optuna VQE trial {trial.number}"
        slot_dir = None
        request_dir = None
        queue_wait_seconds = 0.0
        queue_rank = None
        queue_slot = None
        try:
            queue_slot, slot_dir, request_dir, queue_wait_seconds, queue_rank = acquire_slot(
                queue_root=QUEUE_ROOT,
                max_parallel=args.max_parallel,
                poll_seconds=args.poll_seconds,
                label_key="molecule",
                label_value=args.molecule,
                source_script=str(frozen_source),
                description=description,
                requested_index=trial.number,
            )
            env = os.environ.copy()
            env.update(THREAD_BUDGET_ENV)
            env["AUTORESEARCH_VQE_SNAPSHOT_ROOT"] = str(tracked_snapshot_root)
            env[CONFIG_OVERRIDE_ENV] = json.dumps(config)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "track_iteration.py"),
                    "--script",
                    str(frozen_source),
                    "--molecule",
                    args.molecule,
                    "--wall-seconds",
                    str(args.wall_seconds),
                    "--iteration",
                    str(trial.number),
                    "--description",
                    description,
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            status = "keep"
            error_text = None
            result = None
            if proc.returncode != 0:
                status = "crash"
                error_text = stderr or stdout or "trial evaluation failed"
            else:
                try:
                    result = json.loads(stdout)
                except json.JSONDecodeError as exc:
                    status = "crash"
                    error_text = f"evaluation stdout was not valid JSON: {exc}"

            metadata = {
                "trial": trial.number,
                "molecule": args.molecule,
                "status": status,
                "queue_slot": queue_slot,
                "queue_rank_at_start": queue_rank,
                "queue_wait_seconds": queue_wait_seconds,
                "request_id": request_dir.name if request_dir is not None else None,
                "description": description,
            }
            write_json(trial_dir / "metadata.json", metadata)
            if stdout:
                (trial_dir / "stdout.txt").write_text(stdout + "\n")
            if stderr:
                (trial_dir / "stderr.txt").write_text(stderr + "\n")
            if error_text is not None:
                (trial_dir / "error.txt").write_text(error_text + "\n")

            if result is None:
                penalty_row = {
                    "trial": trial.number,
                    "status": status,
                    "objective": CRASH_PENALTY,
                    "final_energy": CRASH_PENALTY,
                    "target_energy": 0.0,
                    "final_error": CRASH_PENALTY,
                    "chem_acc_step": None,
                    "wall_seconds": args.wall_seconds,
                    "queue_wait_seconds": queue_wait_seconds,
                    "trial_dir": str(trial_dir),
                }
                append_summary(root / "summary.tsv", penalty_row)
                trial.set_user_attr("status", status)
                trial.set_user_attr("trial_dir", str(trial_dir))
                return CRASH_PENALTY

            if result.get("status") == "crash" or result.get("final_energy") is None:
                write_json(trial_dir / "tracker_result.json", result)
                penalty_row = {
                    "trial": trial.number,
                    "status": result.get("status", "crash"),
                    "objective": CRASH_PENALTY,
                    "final_energy": CRASH_PENALTY,
                    "target_energy": 0.0,
                    "final_error": CRASH_PENALTY,
                    "chem_acc_step": None,
                    "wall_seconds": float(result.get("wall_seconds", args.wall_seconds)),
                    "queue_wait_seconds": queue_wait_seconds,
                    "trial_dir": str(trial_dir),
                }
                append_summary(root / "summary.tsv", penalty_row)
                trial.set_user_attr("status", result.get("status", "crash"))
                trial.set_user_attr("trial_dir", str(trial_dir))
                trial.set_user_attr("tracked_snapshot_dir", result.get("snapshot_dir"))
                return CRASH_PENALTY

            tracked_snapshot_dir = Path(result["snapshot_dir"])
            tracked_result_path = tracked_snapshot_dir / "result.json"
            raw_payload = json.loads(tracked_result_path.read_text()) if tracked_result_path.exists() else result
            write_json(trial_dir / "tracker_result.json", result)
            write_json(trial_dir / "result.json", raw_payload)
            row = {
                "trial": trial.number,
                "status": result["status"],
                "objective": float(result["final_energy"]),
                "final_energy": float(result["final_energy"]),
                "target_energy": float(result["target_energy"]),
                "final_error": float(result["final_error"]),
                "chem_acc_step": result.get("chem_acc_step"),
                "wall_seconds": float(result["wall_seconds"]),
                "queue_wait_seconds": queue_wait_seconds,
                "trial_dir": str(trial_dir),
            }
            append_summary(root / "summary.tsv", row)
            trial.set_user_attr("status", result["status"])
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("tracked_snapshot_dir", result.get("snapshot_dir"))
            return float(result["final_energy"])
        finally:
            if slot_dir is not None and slot_dir.exists():
                shutil.rmtree(slot_dir, ignore_errors=True)
            if request_dir is not None and request_dir.exists():
                shutil.rmtree(request_dir, ignore_errors=True)

    finished_trials = [trial for trial in study.trials if trial.value is not None]
    remaining_trials = max(0, args.trials - len(finished_trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)

    write_best_result(root / "best_result.json", study)
    completed_trials = len([trial for trial in study.trials if trial.value is not None])
    best_trials = [trial for trial in study.trials if trial.value is not None]
    best_value = study.best_value if best_trials else None
    print(
        json.dumps(
            {
                "lane": "vqe",
                "molecule": args.molecule,
                "study_root": str(root),
                "frozen_source": str(frozen_source),
                "trials_completed": completed_trials,
                "best_value": best_value,
                "best_trial": study.best_trial.number if best_trials else None,
                "summary_tsv": str(root / "summary.tsv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
