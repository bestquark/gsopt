from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AFQMC_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "periodic_benchmark.py").exists() else SCRIPT_DIR.parent
if str(AFQMC_ROOT) not in sys.path:
    sys.path.insert(0, str(AFQMC_ROOT))

from periodic_benchmark import RunConfig, run_cli

SYSTEM_NAME = "h8_cube_pbc"

DEFAULT_CONFIG = RunConfig(
    name="periodic_h8_cube",
    trial="rhf",
    cell_precision=1e-6,
    conv_tol=1e-5,
    max_cycle=16,
    diis_space=6,
    level_shift=0.10,
    damping=0.20,
    init_guess="atom",
)


if __name__ == "__main__":
    raise SystemExit(run_cli(SYSTEM_NAME, DEFAULT_CONFIG))
