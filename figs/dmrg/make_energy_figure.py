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
from examples.dmrg.model_registry import ACTIVE_MODELS, MODEL_SPECS, PRETTY_LABELS
from examples.dmrg.reference_energies import reference_energy

LANE_DIR = ROOT / "examples" / "dmrg"
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_DMRG_SNAPSHOT_ROOT", LANE_DIR / "snapshots"))
FIG_DIR = Path(os.environ.get("AUTORESEARCH_DMRG_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "dmrg_energy_overview.pdf"
OUTPUT_PNG = FIG_DIR / "dmrg_energy_overview.png"
ORDER = list(ACTIVE_MODELS)


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

    chain_length = MODEL_SPECS[model].chain_length
    frozen_reference = reference_energy(model)
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 6:
            iteration, final_energy, energy_per_site, energy_drop, wall_seconds, _snapshot_dir = parts
            reference_value = None if frozen_reference is None else float(frozen_reference)
            reference_per_site = None if reference_value is None else reference_value / chain_length
            excess_value = None if reference_value is None else float(final_energy) - reference_value
            excess_per_site = None if excess_value is None else excess_value / chain_length
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
                None if reference_value is None else reference_value / chain_length
            )
            excess_value = float(excess_value) if excess_value else (
                None if reference_value is None else float(final_energy) - reference_value
            )
            excess_per_site = float(excess_per_site) if excess_per_site else (
                None if excess_value is None else excess_value / chain_length
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
    ax.axhline(
        machine_precision,
        color="#222222",
        linestyle="--",
        linewidth=2.0,
        zorder=0,
    )
    ylabel = r"Energy Above Best-So-Far per site" if used_fallback else r"Excess energy per site"
    finish_axes(ax, xlabel=r"Iterations", ylabel=ylabel)
    handles.append(
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.0, label="Machine Precision")
    )
    legend = ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#444444")
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
    print(json.dumps({"overview_pdf": str(OUTPUT_PDF), "overview_png": str(OUTPUT_PNG)}, indent=2))


if __name__ == "__main__":
    main()
