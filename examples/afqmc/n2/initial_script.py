from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AFQMC_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "molecular_benchmark.py").exists() else SCRIPT_DIR.parent
if str(AFQMC_ROOT) not in sys.path:
    sys.path.insert(0, str(AFQMC_ROOT))

from molecular_benchmark import RunConfig, run_cli

SYSTEM_NAME = "n2"

DEFAULT_CONFIG = RunConfig(
    name="molecular_n2",
    trial="rhf",
    scf_conv_tol=1e-7,
    scf_max_cycle=64,
    diis_space=8,
    level_shift=0.0,
    damping=0.0,
    init_guess="minao",
    chol_cut=1e-5,
    num_walkers_per_rank=32,
    num_steps_per_block=25,
    num_blocks=40,
    timestep=0.005,
    stabilize_freq=5,
    pop_control_freq=5,
)


if __name__ == "__main__":
    raise SystemExit(run_cli(SYSTEM_NAME, DEFAULT_CONFIG))
