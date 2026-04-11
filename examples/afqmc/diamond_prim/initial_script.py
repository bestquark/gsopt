from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AFQMC_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "periodic_benchmark.py").exists() else SCRIPT_DIR.parent
if str(AFQMC_ROOT) not in sys.path:
    sys.path.insert(0, str(AFQMC_ROOT))

from periodic_benchmark import RunConfig, run_cli

SYSTEM_NAME = "diamond_prim"

DEFAULT_CONFIG = RunConfig(
    name="periodic_diamond_prim",
    trial="rhf",
    cell_precision=1e-7,
    conv_tol=2e-5,
    max_cycle=24,
    diis_space=8,
    level_shift=0.05,
    damping=0.10,
    init_guess="minao",
)


if __name__ == "__main__":
    raise SystemExit(run_cli(SYSTEM_NAME, DEFAULT_CONFIG))
