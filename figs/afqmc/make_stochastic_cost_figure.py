from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes
from examples.afqmc.model_registry import ACTIVE_SYSTEMS

LANE_DIR = ROOT / "examples" / "afqmc"
FIG_DIR = Path(os.environ.get("AUTORESEARCH_AFQMC_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "afqmc_stochastic_cost_over_iterations.pdf"
OUTPUT_PNG = FIG_DIR / "afqmc_stochastic_cost_over_iterations.png"
SYSTEM_COLORS = {
    "h2": "#b14f4f",
    "lih": "#c57b2e",
    "h2o": "#2f8f83",
    "n2": "#375a9e",
}


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage[version=4]{mhchem}"
                r"\usepackage{newtxtext}"
                r"\usepackage{newtxmath}"
            ),
            "font.family": "serif",
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 21,
            "axes.grid": True,
        }
    )


def system_label_tex(system: str) -> str:
    mapping = {
        "h2": r"\ce{H2}",
        "lih": r"\ce{LiH}",
        "h2o": r"\ce{H2O}",
        "n2": r"\ce{N2}",
    }
    return mapping.get(system, system.replace("_", r"\_"))


def _latest_run_dir(system: str) -> Path:
    benchmark_dir = LANE_DIR / system
    runs = sorted(path for path in benchmark_dir.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run_* directory found under {benchmark_dir}")
    return runs[-1]


def _stochastic_cost(result: dict) -> float | None:
    if result.get("status") == "crash":
        return None
    walkers_total = result.get("num_walkers_total")
    if walkers_total is None:
        walkers_per_rank = result.get("num_walkers_per_rank")
        mpi_size = result.get("mpi_size")
        if walkers_per_rank is not None and mpi_size is not None:
            walkers_total = int(walkers_per_rank) * int(mpi_size)
    steps_per_block = result.get("num_steps_per_block")
    num_blocks = result.get("num_blocks_requested")
    if walkers_total is None or steps_per_block is None or num_blocks is None:
        config = result.get("config", {})
        walkers_total = walkers_total if walkers_total is not None else config.get("num_walkers_per_rank")
        steps_per_block = steps_per_block if steps_per_block is not None else config.get("num_steps_per_block")
        num_blocks = num_blocks if num_blocks is not None else config.get("num_blocks")
        mpi_size = result.get("mpi_size", 1)
        if walkers_total is not None and "num_walkers_per_rank" in config:
            walkers_total = int(walkers_total) * int(mpi_size)
    if walkers_total is None or steps_per_block is None or num_blocks is None:
        return None
    return float(int(walkers_total) * int(steps_per_block) * int(num_blocks))


def load_series(system: str) -> tuple[Path, list[int], list[float]]:
    run_dir = _latest_run_dir(system)
    log_path = run_dir / "logs" / "evaluations.jsonl"
    xs: list[int] = []
    ys: list[float] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iteration = row.get("iteration")
        result = row.get("result", {})
        if not isinstance(iteration, int):
            continue
        cost = _stochastic_cost(result)
        if cost is None:
            continue
        xs.append(iteration)
        ys.append(cost)
    return run_dir, xs, ys


def main():
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    runs: dict[str, str] = {}

    for system in ACTIVE_SYSTEMS:
        run_dir, xs, ys = load_series(system)
        runs[system] = str(run_dir)
        if not xs:
            continue
        color = SYSTEM_COLORS.get(system, "#4c4c4c")
        best_so_far: list[float] = []
        incumbent = float("inf")
        for value in ys:
            incumbent = min(incumbent, value)
            best_so_far.append(incumbent)
        ax.scatter(xs, ys, s=62, color=color, alpha=0.24, edgecolors="none", zorder=2)
        ax.plot(xs, best_so_far, color=color, linewidth=3.2, label=system_label_tex(system), zorder=3)

    ax.set_yscale("log")
    finish_axes(
        ax,
        xlabel="Evolution Iteration",
        ylabel=(
            r"Stochastic Cost "
            r"$C_{\mathrm{stoch}} = N_{\mathrm{walkers}} N_{\mathrm{steps}} N_{\mathrm{blocks}}$"
        ),
    )
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_xlim(left=-0.5)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.96)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "overview_pdf": str(OUTPUT_PDF),
                "overview_png": str(OUTPUT_PNG),
                "runs": runs,
                "definition": "C_stoch = N_walkers * N_steps * N_blocks",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
