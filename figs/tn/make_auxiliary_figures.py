from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import annotate_panel, apply_style, finish_axes
from examples.tn.model_registry import PRETTY_LABELS

LANE_DIR = ROOT / "examples" / "tn"
SNAPSHOT_ROOT = LANE_DIR / "snapshots"
FIG_DIR = Path(__file__).resolve().parent
STRUCTURE_PDF = FIG_DIR / "tn_structure_candidates.pdf"
STRUCTURE_PNG = FIG_DIR / "tn_structure_candidates.png"
TRADEOFF_PDF = FIG_DIR / "tn_cost_tradeoffs.pdf"
TRADEOFF_PNG = FIG_DIR / "tn_cost_tradeoffs.png"
LANE_ORDER = [
    "heisenberg_xxx_384",
    "xxz_gapless_256",
    "spin1_heisenberg_64",
    "tfim_2d_4x4",
    "heisenberg_2d_4x4",
]
LANE_MARKERS = {
    "heisenberg_xxx_384": "o",
    "xxz_gapless_256": "s",
    "spin1_heisenberg_64": "^",
    "tfim_2d_4x4": "D",
    "heisenberg_2d_4x4": "P",
}
MARKER_ITERATIONS = [0, 20, 40, 60, 80, 100]


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


def results_rows(lane: str) -> list[dict[str, str]]:
    path = SNAPSHOT_ROOT / lane / "results.tsv"
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row.get("iteration")]


def best_row(lane: str) -> dict[str, str]:
    keeps = [row for row in results_rows(lane) if row.get("status") == "keep" and row.get("final_energy")]
    return min(keeps, key=lambda row: float(row["final_energy"]))


def baseline_row(lane: str) -> dict[str, str]:
    return next(row for row in results_rows(lane) if row["iteration"] == "0")


def snapshot_dir(lane: str, iteration: int) -> Path:
    return SNAPSHOT_ROOT / lane / f"iter_{iteration:04d}"


def snapshot_script(lane: str, iteration: int) -> Path:
    root = snapshot_dir(lane, iteration)
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
            archived_name = payload.get("archived_source_name")
            if archived_name:
                candidate = root / str(archived_name)
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass
    for name in ("initial_script.py",):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing snapshot source file in {root}")


def load_result(lane: str, iteration: int) -> dict:
    return json.loads((snapshot_dir(lane, iteration) / "result.json").read_text())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rerun_history(script_path: Path, wall_seconds: float = 20.0) -> dict:
    module = load_module(script_path, f"_tn_aux_{script_path.parent.name}_{script_path.parent.parent.name}")
    problem = module.build_problem(module.MODEL_NAME)
    return module.run_config(module.DEFAULT_CONFIG, problem, wall_time_limit=wall_seconds)


def draw_grid_structure(ax, title: str, cfg: dict, realized_bond: int, summary_lines: list[str]):
    ax.set_aspect("equal")
    colors = {"checkerboard": ("#d95f02", "#1b9e77"), "plus": ("#4c78a8", "#4c78a8")}
    c0, c1 = colors.get(cfg["init_state"], ("#6c757d", "#adb5bd"))
    for x in range(4):
        for y in range(4):
            if x < 3:
                ax.plot([x, x + 1], [y, y], color="#777777", linewidth=1.4 + 0.35 * realized_bond, alpha=0.65)
            if y < 3:
                ax.plot([x, x], [y, y + 1], color="#777777", linewidth=1.4 + 0.35 * realized_bond, alpha=0.65)
            color = c0 if (x + y) % 2 == 0 else c1
            ax.scatter([x], [y], s=330, color=color, edgecolors="white", linewidths=1.8, zorder=3)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=18)
    annotate_panel(ax, "\n".join(summary_lines))


def draw_chain_structure(ax, title: str, cfg: dict, realized_bond: int, summary_lines: list[str]):
    xs = np.arange(10)
    y = np.zeros_like(xs, dtype=float)
    if cfg["init_state"] == "neel":
        node_colors = ["#d95f02" if i % 2 == 0 else "#1b9e77" for i in range(len(xs))]
    elif cfg["init_state"] == "random":
        node_colors = ["#6a4c93" for _ in xs]
    else:
        node_colors = ["#4c78a8" for _ in xs]
    for i in range(len(xs) - 1):
        ax.plot(
            [xs[i], xs[i + 1]],
            [y[i], y[i + 1]],
            color="#777777",
            linewidth=1.2 + 0.03 * realized_bond,
            alpha=0.72,
            zorder=1,
        )
    ax.scatter(xs, y, s=260, color=node_colors, edgecolors="white", linewidths=1.8, zorder=2)
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=18)
    annotate_panel(ax, "\n".join(summary_lines))


def plot_running_curve(ax, baseline: dict, optimized: dict, reference_final: float, xlabel: str):
    for label, payload, color in [
        ("Initial setup", baseline, "#4c78a8"),
        ("Optimized setup", optimized, "#d95f02"),
    ]:
        history = payload["history"]
        steps = [int(step) for step, _energy, _bond in history]
        deltas = [max(float(energy) - reference_final, 1e-12) for step, energy, _bond in history]
        running = []
        best = float("inf")
        for value in deltas:
            best = min(best, value)
            running.append(best)
        ax.scatter(steps, deltas, s=20, color=color, alpha=0.18)
        ax.step(steps, running, where="post", color=color, linewidth=2.8, label=label)
    ax.set_yscale("log")
    finish_axes(ax, xlabel=xlabel, ylabel=r"$E - E_{\mathrm{best\,final}}$ [arb.]")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#444444")


def make_structure_figure():
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.3], "wspace": 0.28, "hspace": 0.42},
    )

    # 2D candidate
    lane = "heisenberg_2d_4x4"
    base_row = baseline_row(lane)
    best = best_row(lane)
    base_iter = int(base_row["iteration"])
    best_iter = int(best["iteration"])
    base_result = load_result(lane, base_iter)
    best_result = load_result(lane, best_iter)
    base_hist = rerun_history(snapshot_script(lane, base_iter), wall_seconds=20.0)
    best_hist = rerun_history(snapshot_script(lane, best_iter), wall_seconds=20.0)
    draw_grid_structure(
        axes[0, 0],
        "Initial tensor structure",
        base_result["config"],
        int(base_result["max_bond_realized"]),
        [
            r"$4\times4$ Heisenberg",
            r"$\mathrm{TEBD2D}$, checkerboard",
            rf"$D_{{\max}}={base_result['max_bond_realized']}$, $\chi={base_result['config']['chi']}$",
            rf"sweeps {base_result['iterations']}, cutoff {base_result['config']['cutoff']:.0e}",
        ],
    )
    draw_grid_structure(
        axes[0, 1],
        "Optimized tensor structure",
        best_result["config"],
        int(best_result["max_bond_realized"]),
        [
            r"$4\times4$ Heisenberg",
            r"$\mathrm{TEBD2D}$, checkerboard",
            rf"$D_{{\max}}={best_result['max_bond_realized']}$, $\chi={best_result['config']['chi']}$",
            rf"sweeps {best_result['iterations']}, cutoff {best_result['config']['cutoff']:.0e}",
        ],
    )
    plot_running_curve(
        axes[0, 2],
        base_hist,
        best_hist,
        reference_final=float(best_hist["final_energy"]),
        xlabel=r"TEBD Steps",
    )
    axes[0, 2].set_title(r"$4\times4$ Heisenberg internal convergence", fontsize=18)

    # 1D candidate
    lane = "spin1_heisenberg_64"
    base_row = baseline_row(lane)
    best = best_row(lane)
    base_iter = int(base_row["iteration"])
    best_iter = int(best["iteration"])
    base_result = load_result(lane, base_iter)
    best_result = load_result(lane, best_iter)
    base_hist = rerun_history(snapshot_script(lane, base_iter), wall_seconds=20.0)
    best_hist = rerun_history(snapshot_script(lane, best_iter), wall_seconds=20.0)
    draw_chain_structure(
        axes[1, 0],
        "Initial tensor structure",
        base_result["config"],
        int(base_result["max_bond_realized"]),
        [
            r"Spin-1 Heisenberg $L=64$",
            r"$\mathrm{DMRG2}$, N\'eel start",
            rf"$D_{{\max}}={base_result['max_bond_realized']}$, init bond {base_result['config']['init_bond_dim']}",
            rf"sweeps {base_result['iterations']}, cutoff {base_result['config']['cutoff']:.0e}",
        ],
    )
    draw_chain_structure(
        axes[1, 1],
        "Optimized tensor structure",
        best_result["config"],
        int(best_result["max_bond_realized"]),
        [
            r"Spin-1 Heisenberg $L=64$",
            r"$\mathrm{DMRG1}$, random start",
            rf"$D_{{\max}}={best_result['max_bond_realized']}$, init bond {best_result['config']['init_bond_dim']}",
            rf"sweeps {best_result['iterations']}, cutoff {best_result['config']['cutoff']:.0e}",
        ],
    )
    plot_running_curve(
        axes[1, 2],
        base_hist,
        best_hist,
        reference_final=float(best_hist["final_energy"]),
        xlabel=r"DMRG Sweeps",
    )
    axes[1, 2].set_title(r"Spin-1 Heisenberg internal convergence", fontsize=18)

    fig.tight_layout()
    fig.savefig(STRUCTURE_PDF, bbox_inches="tight")
    fig.savefig(STRUCTURE_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_tradeoff_figure():
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(8.8, 11.6),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    cmap = plt.get_cmap("YlGn")
    all_iterations: list[int] = []
    all_rows: dict[str, list[dict[str, str]]] = {}

    for lane in LANE_ORDER:
        rows = [row for row in results_rows(lane) if row.get("final_energy") and row.get("status") != "crash"]
        if not rows:
            continue
        all_rows[lane] = rows
        all_iterations.extend(int(row["iteration"]) for row in rows)

    if not all_iterations:
        ax0.text(0.5, 0.5, "no run data", transform=ax0.transAxes, ha="center", va="center")
        ax1.text(0.5, 0.5, "no run data", transform=ax1.transAxes, ha="center", va="center")
        return

    norm = plt.Normalize(vmin=min(all_iterations), vmax=max(all_iterations))
    handles_lanes = []

    for lane in LANE_ORDER:
        rows = all_rows.get(lane, [])
        if not rows:
            continue
        marker = LANE_MARKERS.get(lane, "o")
        lane_best = min(float(row["final_energy"]) for row in rows)
        best_energy_row = min(rows, key=lambda row: float(row["final_energy"]))
        best_entropy = load_result(lane, int(best_energy_row["iteration"])).get("entropy_midchain")
        xs_energy: list[float] = []
        ys_energy: list[float] = []
        xs_entropy: list[float] = []
        ys_entropy: list[float] = []
        energy_points: list[tuple[int, float, float]] = []
        entropy_points: list[tuple[int, float, float]] = []
        for row in rows:
            iteration = int(row["iteration"])
            result = load_result(lane, iteration)
            energy_above_best = max(float(row["final_energy"]) - lane_best, 1e-12)
            energy_above_best_per_site = energy_above_best / float(result["nsites"])
            x_bond = float(result["max_bond_realized"])
            xs_energy.append(x_bond)
            ys_energy.append(energy_above_best_per_site)
            energy_points.append((iteration, x_bond, energy_above_best_per_site))
            entropy = result.get("entropy_midchain")
            if entropy is not None and best_entropy is not None:
                entropy_gap = max(abs(float(entropy) - float(best_entropy)), 1e-12)
                xs_entropy.append(x_bond)
                ys_entropy.append(entropy_gap)
                entropy_points.append((iteration, x_bond, entropy_gap))

        if len(xs_energy) >= 2:
            ax0.plot(xs_energy, ys_energy, color="#8d8d8d", alpha=0.24, linewidth=1.15, zorder=1)
        if len(xs_entropy) >= 2:
            ax1.plot(xs_entropy, ys_entropy, color="#8d8d8d", alpha=0.24, linewidth=1.15, zorder=1)

        shown_energy_iters: set[int] = set()
        for target in MARKER_ITERATIONS:
            candidates = [point for point in energy_points if point[0] >= target]
            if candidates:
                chosen = min(candidates, key=lambda point: point[0])
            else:
                chosen = energy_points[-1]
            iteration, x_bond, energy_gap = chosen
            if iteration in shown_energy_iters:
                continue
            shown_energy_iters.add(iteration)
            color = cmap(norm(iteration))
            ax0.scatter(
                x_bond,
                energy_gap,
                s=150,
                color=color,
                alpha=0.94,
                marker=marker,
                edgecolors="#2f2f2f",
                linewidths=0.8,
                zorder=2,
            )

        shown_entropy_iters: set[int] = set()
        for target in MARKER_ITERATIONS:
            if not entropy_points:
                break
            candidates = [point for point in entropy_points if point[0] >= target]
            if candidates:
                chosen = min(candidates, key=lambda point: point[0])
            else:
                chosen = entropy_points[-1]
            iteration, x_bond, entropy_gap = chosen
            if iteration in shown_entropy_iters:
                continue
            shown_entropy_iters.add(iteration)
            color = cmap(norm(iteration))
            ax1.scatter(
                x_bond,
                entropy_gap,
                s=150,
                color=color,
                alpha=0.94,
                marker=marker,
                edgecolors="#2f2f2f",
                linewidths=0.8,
                zorder=2,
            )

        handles_lanes.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="#666666",
                markersize=10,
                label=PRETTY_LABELS[lane],
            )
        )

    ax0.set_yscale("log")
    finish_axes(ax0, ylabel=r"Per-site gap [Ha]")

    ax1.set_yscale("log")
    finish_axes(ax1, xlabel=r"Max Bond Realized", ylabel=r"Entropy gap [bits]")

    for ax in (ax0, ax1):
        ax.tick_params(length=6, width=1.1, labelsize=24)

    fig.subplots_adjust(top=0.64, left=0.20, right=0.98, bottom=0.10)
    lane_legend = fig.legend(
        handles=handles_lanes,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
        fontsize=24,
        handlelength=1.0,
        columnspacing=1.0,
        labelspacing=0.55,
        borderpad=0.55,
    )
    lane_legend.get_frame().set_linewidth(1.0)

    cax = fig.add_axes([0.08, 0.80, 0.84, 0.032])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        orientation="horizontal",
        cax=cax,
    )
    cbar.set_label(r"Autoresearch evolutions", fontsize=24)
    cbar.ax.tick_params(labelsize=24, length=5, width=1.0)

    fig.savefig(TRADEOFF_PDF, bbox_inches="tight")
    fig.savefig(TRADEOFF_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_structure_figure()
    make_tradeoff_figure()
    print(
        json.dumps(
            {
                "structure_pdf": str(STRUCTURE_PDF),
                "structure_png": str(STRUCTURE_PNG),
                "tradeoff_pdf": str(TRADEOFF_PDF),
                "tradeoff_png": str(TRADEOFF_PNG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
