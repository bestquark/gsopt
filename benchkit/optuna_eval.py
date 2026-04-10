from __future__ import annotations

import argparse
import json
from pathlib import Path

from optuna_utils import THREAD_BUDGET_ENV, load_python_module, resolve_repo_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate one frozen Optuna trial config.")
    parser.add_argument("--lane", required=True, choices=["vqe", "tn", "afqmc"])
    parser.add_argument("--script", required=True, help="Frozen benchmark script path.")
    parser.add_argument("--config", required=True, help="JSON file containing the RunConfig payload.")
    parser.add_argument("--wall-seconds", required=True, type=float)
    args = parser.parse_args()

    for key, value in THREAD_BUDGET_ENV.items():
        # Keep the scored subprocess consistent with the archived queued evaluators.
        import os

        os.environ[key] = value

    repo_root = Path(__file__).resolve().parents[1]
    script_path = resolve_repo_path(repo_root, args.script)
    config_path = resolve_repo_path(repo_root, args.config)
    config_data = json.loads(config_path.read_text())
    module = load_python_module(script_path, f"optuna_eval_{args.lane}_{script_path.stem}_{script_path.stat().st_mtime_ns}")
    cfg = module.RunConfig(**config_data)

    if args.lane == "vqe":
        problem = module.build_problem(module.MOLECULE_NAME)
        result = module.run_config(
            cfg,
            problem,
            chemical_accuracy=getattr(module, "CHEMICAL_ACCURACY", 1e-3),
            wall_time_limit=args.wall_seconds,
        )
    elif args.lane == "tn":
        problem = module.build_problem(module.MODEL_NAME)
        result = module.run_config(cfg, problem, wall_time_limit=args.wall_seconds)
    else:
        problem = module.build_problem(module.MOLECULE_NAME)
        target_energy = module.reference_energy(module.MOLECULE_NAME)
        if target_energy is None:
            raise SystemExit(f"missing reference energy for {module.MOLECULE_NAME}")
        result = module.run_config(cfg, problem, wall_time_limit=args.wall_seconds, target_energy=target_energy)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
