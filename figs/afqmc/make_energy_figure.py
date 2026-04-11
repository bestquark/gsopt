from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes

from examples.afqmc.model_registry import ACTIVE_SYSTEMS, PRETTY_LABELS
from examples.afqmc.reference_energies import reference_energy

LANE_DIR = ROOT / "examples" / "afqmc"
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_AFQMC_SNAPSHOT_ROOT", LANE_DIR / "snapshots"))
OPTUNA_ROOT = Path(os.environ.get("AUTORESEARCH_AFQMC_OPTUNA_ROOT", LANE_DIR / "optuna"))
FIG_DIR = Path(os.environ.get("AUTORESEARCH_AFQMC_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "afqmc_error_overview.pdf"
OUTPUT_PNG = FIG_DIR / "afqmc_error_overview.png"
COMPARE_PDF = FIG_DIR / "afqmc_error_overview_with_optuna.pdf"
COMPARE_PNG = FIG_DIR / "afqmc_error_overview_with_optuna.png"
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


def load_snapshot_records(system: str) -> list[dict]:
    root = SNAPSHOT_ROOT / system
    if not root.exists():
        return []
    rows = []
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
        result = json.loads(result_path.read_text())
        rows.append({"iteration": iteration, "result": result})
    return rows


def load_optuna_records(system: str) -> list[dict]:
    root = locate_optuna_root(system)
    if root is None:
        return []
    rows = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("trial_"):
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        result = json.loads(result_path.read_text())
        rows.append({"iteration": iteration, "result": result})
    return rows


def locate_optuna_root(system: str) -> Path | None:
    stem = system
    candidates: list[Path] = []
    benchmark_dir = LANE_DIR / stem
    if OPTUNA_ROOT.name.startswith("optuna_run_"):
        candidates.append(OPTUNA_ROOT)
    else:
        candidates.append(OPTUNA_ROOT / stem)
        candidates.append(OPTUNA_ROOT)
    if benchmark_dir.exists():
        candidates.extend(sorted((path for path in benchmark_dir.glob("optuna_run_*") if path.is_dir()), reverse=True))
    for candidate in candidates:
        if candidate.exists() and any(child.is_dir() and child.name.startswith("trial_") for child in candidate.iterdir()):
            return candidate
    return None


def afqmc_error(result: dict, system: str) -> float | None:
    if result.get("abs_final_error") is not None:
        return max(abs(float(result["abs_final_error"])), DELTA_FLOOR)
    if result.get("final_error") is not None:
        return max(abs(float(result["final_error"])), DELTA_FLOOR)
    target = reference_energy(system)
    if target is None:
        return None
    return max(abs(float(result["final_energy"]) - target), DELTA_FLOOR)


def plot_combined(ax):
    cmap = plt.get_cmap("tab10")
    all_errors: list[float] = []
    max_iteration = 1
    min_iteration = 0
    handles: list[Line2D] = []

    for idx, system in enumerate(ACTIVE_SYSTEMS):
        records = load_snapshot_records(system)
        if not records:
            continue
        color = cmap(idx % 10)
        points = [
            (int(record["iteration"]), afqmc_error(record["result"], system))
            for record in records
        ]
        points = [(iteration, error) for iteration, error in points if error is not None]
        if not points:
            continue
        xs = [iteration for iteration, _error in points]
        ys = [error for _iteration, error in points]
        running = []
        best = float("inf")
        for value in ys:
            best = min(best, value)
            running.append(best)
        all_errors.extend(ys)
        min_iteration = min(min_iteration, min(xs))
        max_iteration = max(max_iteration, max(xs))
        ax.scatter(xs, ys, s=34, color=color, alpha=0.22, zorder=1)
        ax.step(xs, running, where="post", color=color, linewidth=3.4, zorder=2)
        handles.append(Line2D([0], [0], color=color, linewidth=3.4, label=PRETTY_LABELS[system]))

    if not all_errors:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return

    ax.set_yscale("log")
    ax.set_xlim(min_iteration, max_iteration if max_iteration > min_iteration else min_iteration + 1)
    ax.set_ylim(min(all_errors) / 1.8, max(all_errors) * 1.8)
    ax.axhline(CHEMICAL_ACCURACY, color="#222222", linestyle="--", linewidth=2.2, zorder=0)
    finish_axes(ax, xlabel=r"Autoresearch evolutions", ylabel=r"$\Delta E$ [Ha]")
    handles.append(Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.2, label="Chemical Accuracy"))
    legend = ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#444444")
    legend.get_frame().set_linewidth(1.0)


def has_optuna_data() -> bool:
    return any(load_optuna_records(system) for system in ACTIVE_SYSTEMS)


def plot_comparison(ax):
    cmap = plt.get_cmap("tab10")
    all_errors: list[float] = []
    max_iteration = 1
    min_iteration = 0
    system_handles: list[Line2D] = []

    for idx, system in enumerate(ACTIVE_SYSTEMS):
        autoresearch_records = load_snapshot_records(system)
        optuna_records = load_optuna_records(system)
        if not autoresearch_records and not optuna_records:
            continue
        color = cmap(idx % 10)

        if autoresearch_records:
            points = [
                (int(record["iteration"]), afqmc_error(record["result"], system))
                for record in autoresearch_records
            ]
            points = [(iteration, error) for iteration, error in points if error is not None]
            xs = [iteration for iteration, _error in points]
            ys = [error for _iteration, error in points]
            running = []
            best = float("inf")
            for value in ys:
                best = min(best, value)
                running.append(best)
            if ys:
                all_errors.extend(ys)
                min_iteration = min(min_iteration, min(xs))
                max_iteration = max(max_iteration, max(xs))
                ax.scatter(xs, ys, s=28, color=color, alpha=0.16, zorder=1)
                ax.step(xs, running, where="post", color=color, linewidth=3.0, zorder=2)

        if optuna_records:
            points = [
                (int(record["iteration"]), afqmc_error(record["result"], system))
                for record in optuna_records
            ]
            points = [(iteration, error) for iteration, error in points if error is not None]
            xs = [iteration for iteration, _error in points]
            ys = [error for _iteration, error in points]
            running = []
            best = float("inf")
            for value in ys:
                best = min(best, value)
                running.append(best)
            if ys:
                all_errors.extend(ys)
                min_iteration = min(min_iteration, min(xs))
                max_iteration = max(max_iteration, max(xs))
                ax.scatter(xs, ys, s=32, marker="x", color=color, alpha=0.34, zorder=3)
                ax.step(xs, running, where="post", color=color, linewidth=2.6, linestyle="--", zorder=4)

        system_handles.append(Line2D([0], [0], color=color, linewidth=3.0, label=PRETTY_LABELS[system]))

    if not all_errors:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return

    ax.set_yscale("log")
    ax.set_xlim(min_iteration, max_iteration if max_iteration > min_iteration else min_iteration + 1)
    ax.set_ylim(min(all_errors) / 1.8, max(all_errors) * 1.8)
    ax.axhline(CHEMICAL_ACCURACY, color="#222222", linestyle="--", linewidth=2.2, zorder=0)
    finish_axes(ax, xlabel=r"Search iterations / trials", ylabel=r"$\Delta E$ [Ha]")
    style_handles = [
        Line2D([0], [0], color="#111111", linewidth=3.0, label="Autoresearch"),
        Line2D([0], [0], color="#111111", linewidth=2.6, linestyle="--", label="Optuna TPE"),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.2, label="Chemical Accuracy"),
    ]
    legend = ax.legend(
        handles=system_handles + style_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
    )
    legend.get_frame().set_linewidth(1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig, ax = plt.subplots(figsize=(13, 8.5))
    plot_combined(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    payload = {"overview_pdf": str(OUTPUT_PDF), "overview_png": str(OUTPUT_PNG)}
    if has_optuna_data():
        fig, ax = plt.subplots(figsize=(13, 8.5))
        plot_comparison(ax)
        fig.tight_layout()
        fig.savefig(COMPARE_PDF, bbox_inches="tight")
        fig.savefig(COMPARE_PNG, dpi=240, bbox_inches="tight")
        plt.close(fig)
        payload["comparison_pdf"] = str(COMPARE_PDF)
        payload["comparison_png"] = str(COMPARE_PNG)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
