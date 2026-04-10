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
from model_registry import ACTIVE_MOLECULES

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
QUEUE_ROOT = SCRIPT_DIR / "eval_queue"
TRACKED_SNAPSHOT_ROOT = SCRIPT_DIR / "optuna_tracked_snapshots"
SUMMARY_HEADER = (
    "trial\tstatus\tobjective\tfinal_energy\treference_energy\tfinal_error\tabs_final_error\t"
    "wall_seconds\tqueue_wait_seconds\ttrial_dir\n"
)
CRASH_PENALTY = 1e9


def default_script_for(molecule: str) -> Path:
    return SCRIPT_DIR / molecule.lower().replace("+", "_plus") / "initial_script.py"


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


def suggest_afqmc_config(trial: optuna.Trial, module) -> dict:
    config = asdict(module.DEFAULT_CONFIG)
    config["name"] = f"optuna_trial_{trial.number:04d}"
    config["trial"] = trial.suggest_categorical("trial", ["rhf", "uhf"])
    config["orbital_basis"] = trial.suggest_categorical("orbital_basis", ["mo", "ortho_ao"])
    config["chol_cut"] = trial.suggest_float("chol_cut", 1e-7, 1e-3, log=True)
    config["timestep"] = trial.suggest_float("timestep", 1e-3, 5e-2, log=True)
    config["num_walkers"] = int(trial.suggest_categorical("num_walkers", [32, 64, 96, 128, 160, 192]))
    config["steps_per_block"] = int(trial.suggest_categorical("steps_per_block", [2, 5, 10, 20]))
    config["stabilize_freq"] = int(trial.suggest_categorical("stabilize_freq", [1, 2, 5, 10, 20]))
    config["pop_control_freq"] = int(trial.suggest_categorical("pop_control_freq", [1, 2, 5, 10, 20]))
    config["max_blocks_cap"] = int(trial.suggest_categorical("max_blocks_cap", [16, 32, 64, 96, 128, 192, 256]))
    config["seed"] = trial.suggest_int("seed", 0, 9999)
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
    parser = argparse.ArgumentParser(description="Run the Optuna AFQMC baseline on a frozen molecule-specific script.")
    parser.add_argument("--script", help="Path to the molecule-specific initial_script.py file.")
    parser.add_argument("--molecule", required=True, choices=ACTIVE_MOLECULES)
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

    source_file = resolve_repo_path(REPO_ROOT, args.script) if args.script else default_script_for(args.molecule)
    root = resolve_repo_path(REPO_ROOT, args.archive_root) if args.archive_root else study_root(SCRIPT_DIR, args.molecule)
    tracked_snapshot_root = root / "tracked_snapshots"
    if args.reset and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    frozen_source, live_hash, frozen_hash = prepare_frozen_source(
        lane_dir=SCRIPT_DIR,
        label=args.molecule,
        source_file=source_file,
        reset=args.reset,
        archive_root=root,
    )
    module = load_python_module(frozen_source, f"afqmc_optuna_source_{slugify(args.molecule)}")

    write_json(
        root / "manifest.json",
        {
            "lane": "afqmc",
            "label": args.molecule,
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
        },
    )

    storage = f"sqlite:///{(root / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=f"afqmc_{slugify(args.molecule)}",
        direction="minimize",
        sampler=build_sampler(args.sampler, args.seed, args.startup_trials),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_afqmc_config(trial, module)
        trial_dir = root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        write_json(trial_dir / "config.json", config)

        description = f"Optuna AFQMC trial {trial.number}"
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
            env["AUTORESEARCH_AFQMC_SNAPSHOT_ROOT"] = str(tracked_snapshot_root)
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

            write_json(
                trial_dir / "metadata.json",
                {
                    "trial": trial.number,
                    "molecule": args.molecule,
                    "status": status,
                    "queue_slot": queue_slot,
                    "queue_rank_at_start": queue_rank,
                    "queue_wait_seconds": queue_wait_seconds,
                    "request_id": request_dir.name if request_dir is not None else None,
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
                        "reference_energy": 0.0,
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

            tracked_snapshot_dir = Path(result["snapshot_dir"])
            tracked_result_path = tracked_snapshot_dir / "result.json"
            raw_payload = json.loads(tracked_result_path.read_text()) if tracked_result_path.exists() else result
            write_json(trial_dir / "tracker_result.json", result)
            write_json(trial_dir / "result.json", raw_payload)
            append_summary(
                root / "summary.tsv",
                {
                    "trial": trial.number,
                    "status": result["status"],
                    "objective": float(result["abs_final_error"]),
                    "final_energy": float(result["final_energy"]),
                    "reference_energy": float(result["reference_energy"]),
                    "final_error": float(result["final_error"]),
                    "abs_final_error": float(result["abs_final_error"]),
                    "wall_seconds": float(result["wall_seconds"]),
                    "queue_wait_seconds": queue_wait_seconds,
                    "trial_dir": str(trial_dir),
                },
            )
            trial.set_user_attr("status", result["status"])
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("tracked_snapshot_dir", result.get("snapshot_dir"))
            return float(result["abs_final_error"])
        finally:
            if slot_dir is not None and slot_dir.exists():
                shutil.rmtree(slot_dir, ignore_errors=True)
            if request_dir is not None and request_dir.exists():
                shutil.rmtree(request_dir, ignore_errors=True)

    remaining_trials = max(0, args.trials - len(study.trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)

    write_best_result(root / "best_result.json", study)
    print(
        json.dumps(
            {
                "lane": "afqmc",
                "molecule": args.molecule,
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
