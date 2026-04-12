from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style
from examples.afqmc.model_registry import ACTIVE_SYSTEMS
from examples.afqmc.reference_energies import reference_energy, reference_method

LANE_DIR = ROOT / "examples" / "afqmc"
FIG_DIR = Path(os.environ.get("AUTORESEARCH_AFQMC_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "afqmc_block_energy_trace_comparison.pdf"
OUTPUT_PNG = FIG_DIR / "afqmc_block_energy_trace_comparison.png"
INITIAL_FILL = "#d8a5a5"
INITIAL_EDGE = "#b14f4f"
OPT_FILL = "#b9d97c"
OPT_EDGE = "#76B900"
REFERENCE_COLOR = "#222222"
CHEMICAL_ACCURACY_COLOR = "#8d8d8d"
CHEMICAL_ACCURACY_HA = 1.0 / 627.509474


@dataclass(frozen=True)
class TraceRecord:
    system: str
    run_dir: Path
    baseline_iteration: int
    best_iteration: int
    reference_label: str
    reference_energy: float | None
    baseline_energy: float
    best_energy: float
    x_label: str
    baseline_x: tuple[float, ...]
    best_x: tuple[float, ...]
    baseline_samples: tuple[float, ...]
    best_samples: tuple[float, ...]


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
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.grid": False,
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _result_from_snapshot(snapshot_dir: Path) -> dict | None:
    result_path = snapshot_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        return _read_json(result_path)
    except json.JSONDecodeError:
        return None


def _result_score(result: dict) -> float | None:
    if result.get("status") == "crash":
        return None
    score = result.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _extract_trace(result: dict) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    tau_grid = result.get("tau_grid")
    for energy_key in ("mixed_estimator_energies", "energy_vs_tau", "walker_energy_trace"):
        energies = result.get(energy_key)
        if isinstance(tau_grid, list) and isinstance(energies, list) and tau_grid and len(tau_grid) == len(energies):
            return (
                r"$\tau$",
                tuple(float(value) for value in tau_grid),
                tuple(float(value) for value in energies),
            )

    values = result.get("block_averaged_energies")
    if isinstance(values, list) and values:
        samples = tuple(float(value) for value in values)
        return ("Block sample index", tuple(float(idx) for idx in range(1, len(samples) + 1)), samples)
    final_energy = float(result["final_energy"])
    return ("Sample index", (1.0,), (final_energy,))


def load_record(system: str) -> TraceRecord:
    run_dir = _latest_run_dir(system)
    baseline_snapshot = run_dir / "snapshots" / "iter_0000"
    baseline_result = _result_from_snapshot(baseline_snapshot)
    if baseline_result is None:
        raise FileNotFoundError(f"missing baseline result.json for {run_dir}")
    best_iteration = 0
    best_result = baseline_result
    best_score = _result_score(baseline_result)
    if best_score is None:
        raise ValueError(f"baseline result for {run_dir} does not contain a valid score")
    for snapshot_dir in sorted((run_dir / "snapshots").glob("iter_*")):
        result = _result_from_snapshot(snapshot_dir)
        if result is None:
            continue
        score = _result_score(result)
        if score is None:
            continue
        if score < best_score:
            best_score = score
            best_result = result
            best_iteration = int(snapshot_dir.name.split("_", 1)[1])
    baseline_x_label, baseline_x, baseline_samples = _extract_trace(baseline_result)
    best_x_label, best_x, best_samples = _extract_trace(best_result)
    x_label = baseline_x_label if baseline_x_label == best_x_label else "Sample index"
    return TraceRecord(
        system=system,
        run_dir=run_dir,
        baseline_iteration=0,
        best_iteration=best_iteration,
        reference_label=reference_method(system) or "Reference",
        reference_energy=reference_energy(system),
        baseline_energy=float(baseline_result["final_energy"]),
        best_energy=float(best_result["final_energy"]),
        x_label=x_label,
        baseline_x=baseline_x,
        best_x=best_x,
        baseline_samples=baseline_samples,
        best_samples=best_samples,
    )


def _running_mean(values: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1, dtype=float)


def main():
    configure_style()
    records = [load_record(system) for system in ACTIVE_SYSTEMS]

    fig, axes = plt.subplots(1, len(records), figsize=(20.0, 5.8), sharey=False)
    if len(records) == 1:
        axes = [axes]

    for axis, record in zip(axes, records):
        baseline_x = np.asarray(record.baseline_x, dtype=float)
        best_x = np.asarray(record.best_x, dtype=float)

        axis.scatter(
            baseline_x,
            record.baseline_samples,
            s=36,
            color=INITIAL_FILL,
            edgecolors="none",
            alpha=0.35,
            zorder=2,
        )
        axis.plot(
            baseline_x,
            _running_mean(record.baseline_samples),
            color=INITIAL_EDGE,
            linewidth=3.0,
            zorder=3,
        )

        axis.scatter(
            best_x,
            record.best_samples,
            s=36,
            color=OPT_FILL,
            edgecolors="none",
            alpha=0.35,
            zorder=2,
        )
        axis.plot(
            best_x,
            _running_mean(record.best_samples),
            color=OPT_EDGE,
            linewidth=3.2,
            zorder=4,
        )

        if record.reference_energy is not None:
            axis.axhspan(
                record.reference_energy - CHEMICAL_ACCURACY_HA,
                record.reference_energy + CHEMICAL_ACCURACY_HA,
                color=CHEMICAL_ACCURACY_COLOR,
                alpha=0.22,
                zorder=0,
            )
            axis.axhline(
                record.reference_energy,
                color=REFERENCE_COLOR,
                linewidth=2.8,
                zorder=1,
            )

        axis.set_title(system_label_tex(record.system), pad=12)
        axis.set_xlabel(record.x_label)
        axis.set_xlim(min(baseline_x.min(), best_x.min()), max(baseline_x.max(), best_x.max()))

        subtitle = (
            rf"Initial $E={record.baseline_energy:.6f}$" "\n"
            rf"Optimized $E={record.best_energy:.6f}$"
        )
        axis.text(
            0.03,
            0.97,
            subtitle,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=17,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#d7d7d7",
                "boxstyle": "round,pad=0.25",
            },
        )

    axes[0].set_ylabel(r"Energy [Ha]")

    legend_handles = [
        Line2D([0], [0], color=INITIAL_EDGE, linewidth=3.0, label="Initial running mean"),
        Line2D([0], [0], color=OPT_EDGE, linewidth=3.2, label="Optimized running mean"),
        Patch(facecolor=CHEMICAL_ACCURACY_COLOR, alpha=0.22, label=r"$\pm 1$ kcal/mol"),
        Line2D([0], [0], color=REFERENCE_COLOR, linewidth=2.8, label=records[-1].reference_label),
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.18, top=0.82, wspace=0.28)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PDF}")
    print(f"wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
