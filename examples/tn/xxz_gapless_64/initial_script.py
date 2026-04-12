from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TN_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "model_registry.py").exists() else SCRIPT_DIR.parent
if str(TN_ROOT) not in sys.path:
    sys.path.insert(0, str(TN_ROOT))
EXAMPLES_ROOT = TN_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from config_override import load_dataclass_override
from simple_tn import RunConfig, build_problem, compact_result, run_config

CONFIG_OVERRIDE_ENV = "AUTORESEARCH_TN_CONFIG_JSON"
MODEL_NAME = "xxz_gapless_64"
SUPPORTED_MODELS = (MODEL_NAME,)

BASELINE_CONFIG = RunConfig(
    name="baseline",
    method="dmrg1",
    init_state="random",
    bond_schedule=(4, 6, 8),
    cutoff=1e-6,
    solver_tol=1e-3,
    max_sweeps=6,
    tau=0.1,
    chi=16,
    init_bond_dim=2,
    init_seed=7,
    local_eig_ncv=4,
)

DEFAULT_CONFIG = RunConfig(
    name="shared_weak_start",
    method="dmrg1",
    init_state="random",
    bond_schedule=(4, 6, 8),
    cutoff=1e-6,
    solver_tol=1e-3,
    max_sweeps=6,
    tau=0.1,
    chi=16,
    init_bond_dim=2,
    init_seed=7,
    local_eig_ncv=4,
)


def runtime_config() -> RunConfig:
    return load_dataclass_override(CONFIG_OVERRIDE_ENV, DEFAULT_CONFIG, RunConfig)


def main():
    parser = argparse.ArgumentParser(description="Run the L=64 critical XXZ tensor-network benchmark.")
    parser.add_argument("--wall-seconds", type=float, default=5.0)
    args = parser.parse_args()

    problem = build_problem(MODEL_NAME)
    result = run_config(runtime_config(), problem, wall_time_limit=args.wall_seconds)
    summary = {
        "task": "simple_tn_ground_state",
        "model": MODEL_NAME,
        "metric": "final_energy",
        "lower_is_better": True,
        "score": result["final_energy"],
        "supported_models": [MODEL_NAME],
        **compact_result(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
