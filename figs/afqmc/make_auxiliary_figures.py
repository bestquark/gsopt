from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes
from examples.afqmc.model_registry import ACTIVE_MOLECULES, PRETTY_LABELS

LANE_DIR = ROOT / "examples" / "afqmc"
SNAPSHOT_ROOT = LANE_DIR / "snapshots"
FIG_DIR = Path(__file__).resolve().parent
TRAJECTORY_PDF = FIG_DIR / "afqmc_cost_tradeoffs.pdf"
TRAJECTORY_PNG = FIG_DIR / "afqmc_cost_tradeoffs.png"
CHEMICAL_ACCURACY = 1e-3
DELTA_FLOOR = 1e-12


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "axes.grid": False,
        }
    )


def load_records(molecule: str) -> list[dict]:
    root = SNAPSHOT_ROOT / molecule.lower().replace("+", "_plus")
    rows = []
    if not root.exists():
        return rows
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        payload = json.loads(result_path.read_text())
        rows.append({"iteration": iteration, "result": payload})
    return rows


def effective_cost(result: dict) -> float:
    cfg = result["config"]
    return float(cfg["num_walkers"]) * float(cfg["steps_per_block"]) * float(result["num_blocks"])


def make_trajectory_figure():
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(8.8, 9.2),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    cmap = plt.get_cmap("tab10")
    handles: list[Line2D] = []
    all_errors: list[float] = []
    all_costs: list[float] = []
    max_iteration = 1

    for idx, molecule in enumerate(ACTIVE_MOLECULES):
        records = load_records(molecule)
        if not records:
            continue
        color = cmap(idx % 10)
        xs = [int(record["iteration"]) for record in records]
        ys = [max(abs(float(record["result"]["final_error"])), DELTA_FLOOR) for record in records]
        costs = [effective_cost(record["result"]) for record in records]
        running = []
        best = float("inf")
        for value in ys:
            best = min(best, value)
            running.append(best)

        max_iteration = max(max_iteration, max(xs))
        all_errors.extend(ys)
        all_costs.extend(costs)

        ax0.scatter(xs, ys, s=34, color=color, alpha=0.22, zorder=1)
        ax0.step(xs, running, where="post", color=color, linewidth=3.4, zorder=2)

        ax1.plot(xs, costs, color=color, linewidth=2.2, alpha=0.85, zorder=2)
        ax1.scatter(xs, costs, s=34, color=color, alpha=0.22, zorder=1)

        handles.append(Line2D([0], [0], color=color, linewidth=3.4, label=PRETTY_LABELS[molecule]))

    if not all_errors or not all_costs:
        ax0.text(0.5, 0.5, "no run data", transform=ax0.transAxes, ha="center", va="center")
        ax1.text(0.5, 0.5, "no run data", transform=ax1.transAxes, ha="center", va="center")
    else:
        ax0.set_yscale("log")
        ax0.set_xlim(0, max_iteration)
        ax0.set_ylim(min(all_errors) / 1.8, max(all_errors) * 1.8)
        ax0.axhline(CHEMICAL_ACCURACY, color="#222222", linestyle="--", linewidth=2.2, zorder=0)
        finish_axes(ax0, ylabel=r"$\Delta E$ [Ha]")

        ax1.set_yscale("log")
        ax1.set_xlim(0, max_iteration)
        ax1.set_ylim(min(all_costs) / 1.5, max(all_costs) * 1.5)
        finish_axes(ax1, xlabel=r"Autoresearch evolutions", ylabel=r"$C_{\mathrm{stoch}}$")

        for ax in (ax0, ax1):
            ax.tick_params(length=6, width=1.1, labelsize=24)

    handles.append(
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.2, label="Chemical Accuracy")
    )
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
        fontsize=24,
        handlelength=2.2,
        columnspacing=1.0,
        labelspacing=0.55,
        borderpad=0.55,
    )
    legend.get_frame().set_linewidth(1.0)

    fig.subplots_adjust(top=0.81, left=0.18, right=0.97, bottom=0.11)
    fig.savefig(TRAJECTORY_PDF, bbox_inches="tight")
    fig.savefig(TRAJECTORY_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_trajectory_figure()
    print(json.dumps({"trajectory_pdf": str(TRAJECTORY_PDF), "trajectory_png": str(TRAJECTORY_PNG)}, indent=2))


if __name__ == "__main__":
    main()
