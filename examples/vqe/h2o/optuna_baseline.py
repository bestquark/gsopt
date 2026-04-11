from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchkit.optuna_wrapper import exec_lane_optuna


if __name__ == "__main__":
    raise SystemExit(
        exec_lane_optuna(
            wrapper_file=__file__,
            lane_script="examples/vqe/optuna_baseline.py",
            source_filename="simple_vqe.py",
            label_flag="--molecule",
            label_value="H2O",
        )
    )
