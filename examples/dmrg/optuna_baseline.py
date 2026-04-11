from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchkit.optuna_utils import (
    load_python_module,
    prepare_frozen_source,
    resolve_repo_path,
    slugify,
    study_root,
    write_json,
)
from benchkit.trial_eval import run_trial_source
from model_registry import ACTIVE_MODELS
from reference_energies import reference_energy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_DMRG_CONFIG_JSON"
SUMMARY_HEADER = (
    "trial\tstatus\tobjective\tfinal_energy\treference_energy\texcess_energy\tenergy_per_site\t"
    "excess_energy_per_site\twall_seconds\tqueue_wait_seconds\ttrial_dir\n"
)
CRASH_PENALTY = 1e9


def default_script_for(model: str) -> Path:
    return SCRIPT_DIR / model / "simple_dmrg.py"


def append_summary(path: Path, row: dict):
    if not path.exists():
        path.write_text(SUMMARY_HEADER)
    reference_value = "" if row["reference_energy"] is None else f"{row['reference_energy']:.12f}"
    excess_energy = "" if row["excess_energy"] is None else f"{row['excess_energy']:.12e}"
    excess_per_site = "" if row["excess_energy_per_site"] is None else f"{row['excess_energy_per_site']:.12e}"
    with path.open("a") as handle:
        handle.write(
            f"{row['trial']}\t{row['status']}\t{row['objective']:.12f}\t{row['final_energy']:.12f}\t"
            f"{reference_value}\t{excess_energy}\t{row['energy_per_site']:.12f}\t"
            f"{excess_per_site}\t{row['wall_seconds']:.4f}\t{row['queue_wait_seconds']:.4f}\t{row['trial_dir']}\n"
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


def _bounded_choices(default: int) -> list[int]:
    values = {
        max(1, default // 2),
        max(1, int(round(default * 0.75))),
        default,
        max(1, int(round(default * 1.25))),
        max(1, default * 2),
    }
    return sorted(values)


def _bond_schedule_choices(default_schedule: list[int]) -> list[int]:
    values: set[int] = set()
    for value in default_schedule:
        values.update({max(2, value // 2), value, max(2, int(round(value * 1.5))), max(2, value * 2)})
    return sorted(values)


def _spin_half_state(name: str) -> tuple[float, float]:
    if name == "up":
        return (1.0, 0.0)
    if name == "down":
        return (0.0, 1.0)
    if name == "plus":
        return (0.7071067811865475, 0.7071067811865475)
    if name == "minus":
        return (0.7071067811865475, -0.7071067811865475)
    raise ValueError(f"unsupported spin-half state {name!r}")


def _spin_one_state(name: str) -> tuple[float, float, float]:
    if name == "up":
        return (1.0, 0.0, 0.0)
    if name == "zero":
        return (0.0, 1.0, 0.0)
    if name == "down":
        return (0.0, 0.0, 1.0)
    if name == "plus":
        scale = 0.5773502691896258
        return (scale, scale, scale)
    raise ValueError(f"unsupported spin-one state {name!r}")


def suggest_dmrg_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    spec = module.MODEL_SPECS[module.MODEL_NAME]
    default_schedule = [int(value) for value in config["bond_schedule"]]
    schedule_choices = _bond_schedule_choices(default_schedule)
    config["name"] = f"optuna_trial_{trial.number:04d}"

    schedule = [int(trial.suggest_categorical(f"bond_dim_{index}", schedule_choices)) for index in range(len(default_schedule))]
    monotonic: list[int] = []
    current = 0
    for value in schedule:
        current = max(current, value)
        monotonic.append(current)
    config["bond_schedule"] = tuple(monotonic)

    config["cutoff"] = trial.suggest_float("cutoff", 1e-14, 1e-6, log=True)
    solver_default = float(config.get("solver_tol", 1e-6))
    if solver_default <= 0.0:
        config["solver_tol"] = float(trial.suggest_categorical("solver_tol", [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]))
    else:
        config["solver_tol"] = trial.suggest_float("solver_tol", 1e-10, 1e-2, log=True)
    if "max_sweeps" in config:
        config["max_sweeps"] = int(trial.suggest_categorical("max_sweeps", _bounded_choices(int(config["max_sweeps"]))))
    if "max_blocks" in config:
        config["max_blocks"] = int(trial.suggest_categorical("max_blocks", _bounded_choices(int(config["max_blocks"]))))
    config["init_bond_dim"] = int(trial.suggest_categorical("init_bond_dim", _bounded_choices(int(config["init_bond_dim"]))))
    config["init_seed"] = int(trial.suggest_int("init_seed", 0, 9999))
    if "stage_sweeps" in config:
        config["stage_sweeps"] = tuple(int(value) for value in config["stage_sweeps"])
    if "sweep_sequence" in config:
        config["sweep_sequence"] = trial.suggest_categorical("sweep_sequence", ["R", "L", "RL", "LR"])
    if "block_sweep_sequence" in config:
        config["block_sweep_sequence"] = trial.suggest_categorical("block_sweep_sequence", ["RL", "LR", "R", "L"])
    if "init_mode" in config:
        config["init_mode"] = trial.suggest_categorical("init_mode", ["random", "product"])
    if "expand_noise" in config:
        config["expand_noise"] = trial.suggest_float("expand_noise", 1e-8, 1e-2, log=True)
    if "local_eig_tol" in config:
        config["local_eig_tol"] = trial.suggest_float("local_eig_tol", 1e-6, 1e-1, log=True)
    if "local_eig_ncv" in config:
        config["local_eig_ncv"] = int(trial.suggest_int("local_eig_ncv", 4, 16))
    if "product_even_state" in config and "product_odd_state" in config:
        if spec.spin == 0.5:
            state_names = ["up", "down", "plus", "minus"]
            config["product_even_state"] = _spin_half_state(trial.suggest_categorical("product_even_state", state_names))
            config["product_odd_state"] = _spin_half_state(trial.suggest_categorical("product_odd_state", state_names))
        else:
            state_names = ["up", "zero", "down", "plus"]
            config["product_even_state"] = _spin_one_state(trial.suggest_categorical("product_even_state", state_names))
            config["product_odd_state"] = _spin_one_state(trial.suggest_categorical("product_odd_state", state_names))
    return config


def write_best_result(path: Path, study: optuna.Study):
    complete_trials = [trial for trial in study.trials if trial.value is not None]
    if not complete_trials:
        return
    best = study.best_trial
    write_json(
        path,
        {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run the Optuna DMRG baseline on a frozen model-specific script.")
    parser.add_argument("--script", help="Path to the model-specific simple_dmrg.py file.")
    parser.add_argument("--model", required=True, choices=ACTIVE_MODELS)
    parser.add_argument("--archive-root", help="Optional directory to store this Optuna run.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--trials", type=int, default=100, help="Target total number of archived Optuna trials.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--sampler", choices=["gp", "tpe", "random"], default="tpe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    source_file = resolve_repo_path(REPO_ROOT, args.script) if args.script else default_script_for(args.model)
    root = resolve_repo_path(REPO_ROOT, args.archive_root) if args.archive_root else study_root(SCRIPT_DIR, args.model)
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    frozen_source, live_hash, frozen_hash = prepare_frozen_source(
        lane_dir=SCRIPT_DIR,
        label=args.model,
        source_file=source_file,
        reset=args.reset,
        archive_root=root,
    )
    module = load_python_module(frozen_source, f"dmrg_optuna_source_{slugify(args.model)}")
    ref_energy = reference_energy(args.model)

    write_json(
        root / "manifest.json",
        {
            "lane": "dmrg",
            "label": args.model,
            "model": module.MODEL_NAME,
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
            "reference_energy": ref_energy,
        },
    )

    storage = f"sqlite:///{(root / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=f"dmrg_{slugify(args.model)}",
        direction="minimize",
        sampler=build_sampler(args.sampler, args.seed, args.startup_trials),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_dmrg_config(trial, module)
        trial_dir = root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        write_json(trial_dir / "config.json", config)

        description = f"Optuna DMRG trial {trial.number}"
        queue_wait_seconds = 0.0
        result, stdout, stderr, error_text = run_trial_source(
            source_file=frozen_source,
            repo_root=REPO_ROOT,
            wall_seconds=args.wall_seconds,
            extra_env={CONFIG_OVERRIDE_ENV: json.dumps(config)},
        )

        status = "keep" if result is not None else "crash"
        write_json(
            trial_dir / "metadata.json",
            {
                "trial": trial.number,
                "model": args.model,
                "status": status,
                "queue_wait_seconds": queue_wait_seconds,
                "description": description,
            },
        )
        if stdout:
            (trial_dir / "stdout.txt").write_text(stdout + "\n")
        if stderr:
            (trial_dir / "stderr.txt").write_text(stderr + "\n")
        if error_text is not None:
            (trial_dir / "error.txt").write_text(error_text + "\n")

        if result is None:
            append_summary(
                root / "summary.tsv",
                {
                    "trial": trial.number,
                    "status": status,
                    "objective": CRASH_PENALTY,
                    "final_energy": CRASH_PENALTY,
                    "reference_energy": ref_energy,
                    "excess_energy": None if ref_energy is None else CRASH_PENALTY,
                    "energy_per_site": CRASH_PENALTY,
                    "excess_energy_per_site": None if ref_energy is None else CRASH_PENALTY,
                    "wall_seconds": args.wall_seconds,
                    "queue_wait_seconds": queue_wait_seconds,
                    "trial_dir": str(trial_dir),
                },
            )
            raise optuna.TrialPruned(error_text or "trial crashed")

        final_energy = float(result["final_energy"])
        excess_energy = None if ref_energy is None else final_energy - ref_energy
        objective_value = final_energy
        energy_per_site = float(result["energy_per_site"])
        chain_length = int(result["chain_length"])
        excess_per_site = None if excess_energy is None else excess_energy / chain_length

        write_json(trial_dir / "result.json", result)
        trial.set_user_attr("status", result.get("status", "keep"))
        trial.set_user_attr("final_energy", final_energy)
        trial.set_user_attr("excess_energy", excess_energy)
        append_summary(
            root / "summary.tsv",
            {
                "trial": trial.number,
                "status": result.get("status", "keep"),
                "objective": objective_value,
                "final_energy": final_energy,
                "reference_energy": ref_energy,
                "excess_energy": excess_energy,
                "energy_per_site": energy_per_site,
                "excess_energy_per_site": excess_per_site,
                "wall_seconds": float(result["wall_seconds"]),
                "queue_wait_seconds": queue_wait_seconds,
                "trial_dir": str(trial_dir),
            },
        )
        return objective_value

    completed = sum(1 for trial in study.trials if trial.state.is_finished())
    while completed < args.trials:
        study.optimize(objective, n_trials=1, catch=(optuna.TrialPruned,))
        completed = sum(1 for trial in study.trials if trial.state.is_finished())
        write_best_result(root / "best_result.json", study)

    write_best_result(root / "best_result.json", study)
    print(
        json.dumps(
            {
                "archive_root": str(root),
                "best_trial": study.best_trial.number if study.best_trial else None,
                "best_value": study.best_value if study.best_trial else None,
                "completed_trials": completed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
