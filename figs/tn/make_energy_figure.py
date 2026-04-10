from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes
from examples.tn.model_registry import ACTIVE_MODELS, MODEL_SPECS, PRETTY_LABELS
from examples.tn.reference_energies import reference_energy

LANE_DIR = ROOT / "examples" / "tn"
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_TN_SNAPSHOT_ROOT", LANE_DIR / "snapshots"))
OPTUNA_ROOT = Path(os.environ.get("AUTORESEARCH_TN_OPTUNA_ROOT", LANE_DIR / "optuna"))
FIG_DIR = Path(os.environ.get("AUTORESEARCH_TN_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "tn_energy_overview.pdf"
OUTPUT_PNG = FIG_DIR / "tn_energy_overview.png"
COMPARE_PDF = FIG_DIR / "tn_energy_overview_with_optuna.pdf"
COMPARE_PNG = FIG_DIR / "tn_energy_overview_with_optuna.png"
ORDER = list(ACTIVE_MODELS)


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 28,
            "axes.labelsize": 28,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 24,
        }
    )


def pretty_label(model: str) -> str:
    return PRETTY_LABELS[model]


def load_snapshot_records(model: str) -> list[dict]:
    summary_path = SNAPSHOT_ROOT / model / "summary.tsv"
    if not summary_path.exists():
        return []
    records = []
    lines = [line.rstrip("\n") for line in summary_path.read_text().splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    nsites = MODEL_SPECS[model].nsites
    frozen_reference = reference_energy(model)
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 6:
            iteration, final_energy, energy_per_site, energy_drop, wall_seconds, _snapshot_dir = parts
            reference_value = None if frozen_reference is None else float(frozen_reference)
            reference_per_site = None if reference_value is None else reference_value / nsites
            excess_value = None if reference_value is None else float(final_energy) - reference_value
            excess_per_site = None if excess_value is None else excess_value / nsites
        elif len(parts) >= 10:
            (
                iteration,
                final_energy,
                reference_value,
                excess_value,
                energy_per_site,
                reference_per_site,
                excess_per_site,
                energy_drop,
                wall_seconds,
                _snapshot_dir,
                *_rest,
            ) = parts
            reference_value = float(reference_value) if reference_value else (
                None if frozen_reference is None else float(frozen_reference)
            )
            reference_per_site = float(reference_per_site) if reference_per_site else (
                None if reference_value is None else reference_value / nsites
            )
            excess_value = float(excess_value) if excess_value else (
                None if reference_value is None else float(final_energy) - reference_value
            )
            excess_per_site = float(excess_per_site) if excess_per_site else (
                None if excess_value is None else excess_value / nsites
            )
        else:
            continue

        records.append(
            {
                "iteration": int(iteration),
                "final_energy": float(final_energy),
                "energy_per_site": float(energy_per_site),
                "reference_energy": reference_value,
                "reference_energy_per_site": reference_per_site,
                "excess_energy": excess_value,
                "excess_energy_per_site": excess_per_site,
                "energy_drop": float(energy_drop),
                "wall_seconds": float(wall_seconds),
            }
        )
    return records


def load_optuna_records(model: str) -> list[dict]:
    root = locate_optuna_root(model)
    if root is None:
        return []
    records = []
    nsites = MODEL_SPECS[model].nsites
    frozen_reference = reference_energy(model)
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
        reference_value = None if frozen_reference is None else float(frozen_reference)
        excess_value = None if reference_value is None else float(result["final_energy"]) - reference_value
        records.append(
            {
                "iteration": iteration,
                "final_energy": float(result["final_energy"]),
                "energy_per_site": float(result["energy_per_site"]),
                "reference_energy": reference_value,
                "reference_energy_per_site": None if reference_value is None else reference_value / nsites,
                "excess_energy": excess_value,
                "excess_energy_per_site": None if excess_value is None else excess_value / nsites,
                "energy_drop": float(result["energy_drop"]),
                "wall_seconds": float(result["wall_seconds"]),
            }
        )
    return records


def locate_optuna_root(model: str) -> Path | None:
    candidates: list[Path] = []
    benchmark_dir = LANE_DIR / model
    if OPTUNA_ROOT.name.startswith("optuna_run_"):
        candidates.append(OPTUNA_ROOT)
    else:
        candidates.append(OPTUNA_ROOT / model)
        candidates.append(OPTUNA_ROOT)
    if benchmark_dir.exists():
        candidates.extend(sorted((path for path in benchmark_dir.glob("optuna_run_*") if path.is_dir()), reverse=True))
    for candidate in candidates:
        if candidate.exists() and any(child.is_dir() and child.name.startswith("trial_") for child in candidate.iterdir()):
            return candidate
    return None


def load_records(model: str) -> list[dict]:
    return load_snapshot_records(model)


def plot_combined(ax):
    cmap = plt.get_cmap("tab10")
    all_values: list[float] = []
    max_iteration = 1
    min_iteration = 0
    handles: list[Line2D] = []
    used_fallback = False
    machine_precision = np.finfo(float).eps

    for idx, model in enumerate(ORDER):
        records = load_records(model)
        if not records:
            continue

        color = cmap(idx % 10)
        xs = [int(record["iteration"]) for record in records]
        if all(record.get("excess_energy_per_site") is not None for record in records):
            raw_values = [float(record["excess_energy_per_site"]) for record in records]
        else:
            best_energy_per_site = min(float(record["energy_per_site"]) for record in records)
            raw_values = [float(record["energy_per_site"]) - best_energy_per_site for record in records]
            used_fallback = True

        ys = [max(value, 1e-16) for value in raw_values]
        running = []
        best = float("inf")
        for value in ys:
            best = min(best, value)
            running.append(best)

        all_values.extend(ys)
        min_iteration = min(min_iteration, min(xs))
        max_iteration = max(max_iteration, max(xs))
        ax.scatter(xs, ys, s=28, color=color, alpha=0.18, zorder=1)
        ax.step(xs, running, where="post", color=color, linewidth=3.0, zorder=2)
        handles.append(Line2D([0], [0], color=color, linewidth=2.6, label=pretty_label(model)))

    if not all_values:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return

    ymin = min(all_values)
    ymax = max(all_values)
    ax.set_yscale("log")
    ax.set_ylim(max(1e-16, min(ymin, machine_precision) / 1.8), ymax * 1.8)
    ax.set_xlim(min_iteration, max_iteration)
    ax.axhline(machine_precision, color="#222222", linestyle="--", linewidth=2.0, zorder=0)
    ylabel = r"Per-site energy gap [Ha]"
    finish_axes(ax, xlabel=r"Autoresearch evolutions", ylabel=ylabel)
    handles.append(Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.0, label="Machine Precision"))
    legend = ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#444444")
    legend.get_frame().set_linewidth(1.0)


def has_optuna_data() -> bool:
    return any(load_optuna_records(model) for model in ORDER)


def plot_comparison(ax):
    cmap = plt.get_cmap("tab10")
    all_values: list[float] = []
    max_iteration = 1
    min_iteration = 0
    molecule_handles: list[Line2D] = []
    machine_precision = np.finfo(float).eps

    for idx, model in enumerate(ORDER):
        autoresearch_records = load_records(model)
        optuna_records = load_optuna_records(model)
        if not autoresearch_records and not optuna_records:
            continue

        color = cmap(idx % 10)
        if autoresearch_records:
            xs = [int(record["iteration"]) for record in autoresearch_records]
            ys = [
                max(
                    float(record["excess_energy_per_site"])
                    if record.get("excess_energy_per_site") is not None
                    else float(record["energy_per_site"]) - min(float(row["energy_per_site"]) for row in autoresearch_records),
                    1e-16,
                )
                for record in autoresearch_records
            ]
            running = []
            best = float("inf")
            for value in ys:
                best = min(best, value)
                running.append(best)
            all_values.extend(ys)
            min_iteration = min(min_iteration, min(xs))
            max_iteration = max(max_iteration, max(xs))
            ax.scatter(xs, ys, s=26, color=color, alpha=0.14, zorder=1)
            ax.step(xs, running, where="post", color=color, linewidth=3.0, zorder=2)

        if optuna_records:
            xs = [int(record["iteration"]) for record in optuna_records]
            ys = [
                max(
                    float(record["excess_energy_per_site"])
                    if record.get("excess_energy_per_site") is not None
                    else float(record["energy_per_site"]) - min(float(row["energy_per_site"]) for row in optuna_records),
                    1e-16,
                )
                for record in optuna_records
            ]
            running = []
            best = float("inf")
            for value in ys:
                best = min(best, value)
                running.append(best)
            all_values.extend(ys)
            min_iteration = min(min_iteration, min(xs))
            max_iteration = max(max_iteration, max(xs))
            ax.scatter(xs, ys, s=30, marker="x", color=color, alpha=0.32, zorder=3)
            ax.step(xs, running, where="post", color=color, linewidth=2.6, linestyle="--", zorder=4)

        molecule_handles.append(Line2D([0], [0], color=color, linewidth=2.8, label=pretty_label(model)))

    if not all_values:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return

    ymin = min(all_values)
    ymax = max(all_values)
    ax.set_yscale("log")
    ax.set_ylim(max(1e-16, min(ymin, machine_precision) / 1.8), ymax * 1.8)
    ax.set_xlim(min_iteration, max_iteration if max_iteration > min_iteration else min_iteration + 1)
    ax.axhline(machine_precision, color="#222222", linestyle="--", linewidth=2.0, zorder=0)
    finish_axes(ax, xlabel=r"Search iterations / trials", ylabel=r"Per-site energy gap [Ha]")
    style_handles = [
        Line2D([0], [0], color="#111111", linewidth=3.0, label="Autoresearch"),
        Line2D([0], [0], color="#111111", linewidth=2.6, linestyle="--", label="Optuna TPE"),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.0, label="Machine Precision"),
    ]
    legend = ax.legend(
        handles=molecule_handles + style_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
    )
    legend.get_frame().set_linewidth(1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig, ax = plt.subplots(figsize=(16, 11))
    plot_combined(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    payload = {"overview_pdf": str(OUTPUT_PDF), "overview_png": str(OUTPUT_PNG)}
    if has_optuna_data():
        fig, ax = plt.subplots(figsize=(16, 11))
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
