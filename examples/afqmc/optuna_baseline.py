from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchkit.optuna_utils import load_python_module, prepare_frozen_source, resolve_repo_path, slugify, study_root, write_json
from benchkit.trial_eval import run_trial_source

try:
    from .model_registry import ACTIVE_SYSTEMS
    from .reference_energies import reference_energy
except ImportError:
    from model_registry import ACTIVE_SYSTEMS
    from reference_energies import reference_energy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_OVERRIDE_ENV = "AUTORESEARCH_AFQMC_CONFIG_JSON"
SUMMARY_HEADER = (
    "trial\tstatus\tobjective\tfinal_energy\treference_energy\tfinal_error\tabs_final_error\t"
    "wall_seconds\tqueue_wait_seconds\ttrial_dir\n"
)
CRASH_PENALTY = 1e9


def default_script_for(system: str) -> Path:
    return SCRIPT_DIR / system / "initial_script.py"


def append_summary(path: Path, row: dict):
    if not path.exists():
        path.write_text(SUMMARY_HEADER)
    with path.open("a") as handle:
        handle.write(
            f"{row['trial']}\t{row['status']}\t{row['objective']:.12e}\t{row['final_energy']:.12f}\t"
            f"{row['reference_energy']:.12f}\t{row['final_error']:.12e}\t{row['abs_final_error']:.12e}\t"
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


def suggest_periodic_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["trial"] = trial.suggest_categorical("trial", ["rhf", "uhf"])
    config["cell_precision"] = trial.suggest_float("cell_precision", 1e-9, 1e-5, log=True)
    config["conv_tol"] = trial.suggest_float("conv_tol", 1e-9, 1e-4, log=True)
    config["max_cycle"] = int(trial.suggest_categorical("max_cycle", [8, 12, 16, 24, 32, 48, 64, 96]))
    config["diis_space"] = int(trial.suggest_categorical("diis_space", [4, 6, 8, 10, 12]))
    config["level_shift"] = float(trial.suggest_categorical("level_shift", [0.0, 0.01, 0.05, 0.10, 0.20]))
    config["damping"] = float(trial.suggest_categorical("damping", [0.0, 0.05, 0.10, 0.20, 0.30]))
    config["init_guess"] = trial.suggest_categorical("init_guess", ["minao", "atom", "1e"])
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
    parser = argparse.ArgumentParser(description="Run the Optuna periodic-electronic baseline on a frozen system-specific script.")
    parser.add_argument("--script", help="Path to the periodic-system initial_script.py file.")
    parser.add_argument("--system", required=True, choices=ACTIVE_SYSTEMS)
    parser.add_argument("--archive-root", help="Optional directory to store this Optuna run.")
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    parser.add_argument("--trials", type=int, default=100, help="Target total number of archived Optuna trials.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--sampler", choices=["gp", "tpe", "random"], default="tpe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    source_file = resolve_repo_path(REPO_ROOT, args.script) if args.script else default_script_for(args.system)
    root = resolve_repo_path(REPO_ROOT, args.archive_root) if args.archive_root else study_root(SCRIPT_DIR, args.system)
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    frozen_source, live_hash, frozen_hash = prepare_frozen_source(
        lane_dir=SCRIPT_DIR,
        label=args.system,
        source_file=source_file,
        reset=args.reset,
        archive_root=root,
    )
    module = load_python_module(frozen_source, f"afqmc_periodic_optuna_source_{slugify(args.system)}")
    target_energy = reference_energy(args.system)
    if target_energy is None:
        raise SystemExit(f"missing reference energy for {args.system}; run compute_reference_energies.py first")

    write_json(
        root / "manifest.json",
        {
            "lane": "afqmc",
            "label": args.system,
            "system": args.system,
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
            "reference_energy": target_energy,
        },
    )

    storage = f"sqlite:///{(root / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=f"afqmc_{slugify(args.system)}",
        direction="minimize",
        sampler=build_sampler(args.sampler, args.seed, args.startup_trials),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_periodic_config(trial, module)
        trial_dir = root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        write_json(trial_dir / "config.json", config)

        description = f"Optuna periodic trial {trial.number}"
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
                "system": args.system,
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
                    "reference_energy": target_energy,
                    "final_error": CRASH_PENALTY,
                    "abs_final_error": CRASH_PENALTY,
                    "wall_seconds": args.wall_seconds,
                    "queue_wait_seconds": queue_wait_seconds,
                    "trial_dir": str(trial_dir),
                },
            )
            trial.set_user_attr("status", status)
            trial.set_user_attr("trial_dir", str(trial_dir))
            return CRASH_PENALTY

        write_json(trial_dir / "result.json", result)
        append_summary(
            root / "summary.tsv",
                {
                    "trial": trial.number,
                    "status": result.get("status", "keep"),
                    "objective": float(result["final_energy"]),
                    "final_energy": float(result["final_energy"]),
                    "reference_energy": float(result.get("reference_energy", target_energy)),
                    "final_error": float(result["final_error"]),
                    "abs_final_error": float(result["abs_final_error"]),
                    "wall_seconds": float(result["wall_seconds"]),
                    "queue_wait_seconds": queue_wait_seconds,
                    "trial_dir": str(trial_dir),
            },
        )
        trial.set_user_attr("status", result.get("status", "keep"))
        trial.set_user_attr("trial_dir", str(trial_dir))
        return float(result["final_energy"])

    remaining_trials = max(0, args.trials - len(study.trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)

    write_best_result(root / "best_result.json", study)
    print(
        json.dumps(
            {
                "lane": "afqmc",
                "system": args.system,
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
