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
from model_registry import AVAILABLE_MODELS
from reference_energies import reference_energy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_TN_CONFIG_JSON"
SUMMARY_HEADER = (
    "trial\tstatus\tobjective\tfinal_energy\treference_energy\texcess_energy\tenergy_per_site\t"
    "excess_energy_per_site\twall_seconds\tqueue_wait_seconds\ttrial_dir\n"
)
CRASH_PENALTY = 1e9


def default_script_for(model: str) -> Path:
    return SCRIPT_DIR / model / "initial_script.py"


def append_summary(path: Path, row: dict):
    if not path.exists():
        path.write_text(SUMMARY_HEADER)
    reference_energy = "" if row["reference_energy"] is None else f"{row['reference_energy']:.12f}"
    excess_energy = "" if row["excess_energy"] is None else f"{row['excess_energy']:.12e}"
    excess_energy_per_site = "" if row["excess_energy_per_site"] is None else f"{row['excess_energy_per_site']:.12e}"
    with path.open("a") as handle:
        handle.write(
            f"{row['trial']}\t{row['status']}\t{row['objective']:.12f}\t{row['final_energy']:.12f}\t"
            f"{reference_energy}\t{excess_energy}\t{row['energy_per_site']:.12f}\t"
            f"{excess_energy_per_site}\t{row['wall_seconds']:.4f}\t"
            f"{row['queue_wait_seconds']:.4f}\t{row['trial_dir']}\n"
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


def init_state_choices(spec) -> list[str]:
    choices = ["product_up", "product_down", "plus", "neel", "checkerboard", "random"]
    if spec.spin == 0.5:
        choices.insert(3, "minus")
    if spec.spin == 1.0:
        choices.insert(3, "zero")
    return choices


def build_bond_schedule(trial: optuna.Trial, spec) -> tuple[int, ...]:
    if spec.geometry == "2d":
        start_choices = [2, 4, 6, 8, 10, 12]
        cap_choices = [8, 12, 16, 24, 32, 48, 64]
    else:
        start_choices = [4, 8, 12, 16, 24, 32]
        cap_choices = [32, 48, 64, 80, 96, 128, 160, 192, 256]
    start = int(trial.suggest_categorical("bond_start", start_choices))
    growth = float(trial.suggest_categorical("bond_growth", [1.25, 1.5, 1.75, 2.0]))
    stages = int(trial.suggest_int("bond_stages", 3, 8))
    cap = int(trial.suggest_categorical("bond_cap", cap_choices))
    values: list[int] = []
    current = start
    for _ in range(stages):
        values.append(max(2, min(cap, int(round(current)))))
        current = max(current + 1, int(round(current * growth)))
    deduped: list[int] = []
    for value in values:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    return tuple(deduped)


def suggest_tn_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    spec = module.MODEL_SPECS[module.MODEL_NAME]
    if spec.geometry == "1d":
        config["method"] = trial.suggest_categorical("method", ["dmrg1", "dmrg2", "tebd1d"])
    else:
        config["method"] = "tebd2d"
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["init_state"] = trial.suggest_categorical("init_state", init_state_choices(spec))
    config["bond_schedule"] = build_bond_schedule(trial, spec)
    config["cutoff"] = trial.suggest_float("cutoff", 1e-14, 1e-6, log=True)
    config["solver_tol"] = trial.suggest_float("solver_tol", 1e-10, 1e-2, log=True)
    config["max_sweeps"] = int(trial.suggest_categorical("max_sweeps", [4, 8, 12, 16, 24, 32, 48, 64]))
    config["tau"] = trial.suggest_float("tau", 1e-3, 5e-1, log=True)
    config["chi"] = int(trial.suggest_categorical("chi", [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]))
    config["init_bond_dim"] = int(trial.suggest_categorical("init_bond_dim", [1, 2, 4, 8, 12, 16, 24, 32]))
    config["init_seed"] = trial.suggest_int("init_seed", 0, 9999)
    config["local_eig_ncv"] = int(trial.suggest_int("local_eig_ncv", 4, 16))
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
    parser = argparse.ArgumentParser(description="Run the Optuna TN baseline on a frozen model-specific script.")
    parser.add_argument("--script", help="Path to the model-specific initial_script.py file.")
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
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
    module = load_python_module(frozen_source, f"tn_optuna_source_{slugify(args.model)}")
    ref_energy = reference_energy(args.model)

    write_json(
        root / "manifest.json",
        {
            "lane": "tn",
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
        study_name=f"tn_{slugify(args.model)}",
        direction="minimize",
        sampler=build_sampler(args.sampler, args.seed, args.startup_trials),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_tn_config(trial, module)
        trial_dir = root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        write_json(trial_dir / "config.json", config)

        description = f"Optuna TN trial {trial.number}"
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
            trial.set_user_attr("status", status)
            trial.set_user_attr("trial_dir", str(trial_dir))
            return CRASH_PENALTY

        write_json(trial_dir / "result.json", result)
        nsites = int(result["nsites"])
        final_energy = float(result["final_energy"])
        energy_per_site = float(result["energy_per_site"])
        excess_energy = None if ref_energy is None else final_energy - ref_energy
        excess_energy_per_site = None if excess_energy is None else excess_energy / nsites
        append_summary(
            root / "summary.tsv",
            {
                "trial": trial.number,
                "status": result.get("status", "keep"),
                "objective": final_energy,
                "final_energy": final_energy,
                "reference_energy": ref_energy,
                "excess_energy": excess_energy,
                "energy_per_site": energy_per_site,
                "excess_energy_per_site": excess_energy_per_site,
                "wall_seconds": float(result["wall_seconds"]),
                "queue_wait_seconds": queue_wait_seconds,
                "trial_dir": str(trial_dir),
            },
        )
        trial.set_user_attr("status", result.get("status", "keep"))
        trial.set_user_attr("trial_dir", str(trial_dir))
        return final_energy

    remaining_trials = max(0, args.trials - len(study.trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)

    write_best_result(root / "best_result.json", study)
    print(
        json.dumps(
            {
                "lane": "tn",
                "model": args.model,
                "study_root": str(root),
                "frozen_source": str(frozen_source),
                "trials_completed": len(study.trials),
                "best_value": study.best_value if study.trials else None,
                "best_trial": study.best_trial.number if study.trials else None,
                "summary_tsv": str(root / "summary.tsv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
