from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
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
RUN_SELECTION = os.environ.get("AUTORESEARCH_AFQMC_RUN_SELECTION", "completed").strip().lower()
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
    baseline_params: str
    best_params: str
    baseline_tail_start: float | None
    best_tail_start: float | None


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
    override = os.environ.get(f"AUTORESEARCH_AFQMC_RUN_OVERRIDE_{system.upper()}")
    if override:
        return Path(override).expanduser().resolve()
    benchmark_dir = LANE_DIR / system
    runs = sorted(path for path in benchmark_dir.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run_* directory found under {benchmark_dir}")
    if RUN_SELECTION == "latest":
        return runs[-1]
    completed: list[Path] = []
    for run_dir in runs:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            status = _read_json(status_path)
        except json.JSONDecodeError:
            continue
        if status.get("done") is True:
            completed.append(run_dir)
    if completed:
        return completed[-1]
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


def _tail_start(result: dict, x_label: str, x_values: tuple[float, ...]) -> float | None:
    production_tau = result.get("production_tau_grid")
    if x_label == r"$\tau$" and isinstance(production_tau, list) and production_tau:
        return float(production_tau[0])

    discarded_blocks = result.get("discarded_block_count", result.get("equilibration_block_count"))
    if not isinstance(discarded_blocks, int) or discarded_blocks <= 0:
        return None

    if x_label == "Block sample index":
        return float(discarded_blocks) + 0.5

    if x_values and len(x_values) >= discarded_blocks:
        return float(x_values[discarded_blocks - 1])
    return None


def _format_run_params(result: dict) -> str:
    config = result.get("config", {})
    timestep = config.get("timestep", result.get("timestep"))
    walkers_total = result.get("num_walkers_total")
    if walkers_total is None:
        walkers = config.get("num_walkers_per_rank")
        mpi_size = result.get("mpi_size")
        if walkers is not None and mpi_size is not None:
            walkers_total = int(walkers) * int(mpi_size)
    steps_per_block = config.get("num_steps_per_block", result.get("num_steps_per_block"))
    num_blocks = config.get("num_blocks", result.get("num_blocks_requested"))
    stabilize = config.get("stabilize_freq", result.get("stabilize_freq"))
    pop_control = config.get("pop_control_freq", result.get("pop_control_freq"))
    return "\n".join(
        [
            rf"$\tau = {float(timestep):g}$",
            rf"$N_{{\mathrm{{w}}}} = {int(walkers_total)}$",
            rf"$B = {int(steps_per_block)} \times {int(num_blocks)}$",
            rf"$s/p = {int(stabilize)} / {int(pop_control)}$",
        ]
    )


def _add_param_box(axis, x: float, y: float, title: str, body: str):
    title_area = TextArea(
        title,
        textprops={
            "fontsize": 14,
            "ha": "center",
            "va": "bottom",
        },
    )
    body_area = TextArea(
        body,
        textprops={
            "fontsize": 14,
            "ha": "left",
            "va": "bottom",
            "multialignment": "left",
        },
    )
    box = VPacker(children=[title_area, body_area], align="center", pad=0, sep=2)
    annotation = AnnotationBbox(
        box,
        (x, y),
        xycoords="axes fraction",
        box_alignment=(0.0, 1.0),
        frameon=True,
        bboxprops={
            "facecolor": "white",
            "alpha": 0.94,
            "edgecolor": "#d7d7d7",
            "boxstyle": "round,pad=0.22",
        },
        zorder=5,
    )
    axis.add_artist(annotation)


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
        baseline_params=_format_run_params(baseline_result),
        best_params=_format_run_params(best_result),
        baseline_tail_start=_tail_start(baseline_result, x_label, baseline_x),
        best_tail_start=_tail_start(best_result, x_label, best_x),
    )


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
            s=28,
            color=INITIAL_FILL,
            edgecolors="none",
            alpha=0.35,
            zorder=2,
        )
        axis.plot(
            baseline_x,
            record.baseline_samples,
            color=INITIAL_EDGE,
            linewidth=1.5,
            alpha=0.85,
            zorder=3,
        )

        axis.scatter(
            best_x,
            record.best_samples,
            s=28,
            color=OPT_FILL,
            edgecolors="none",
            alpha=0.35,
            zorder=2,
        )
        axis.plot(
            best_x,
            record.best_samples,
            color=OPT_EDGE,
            linewidth=1.6,
            alpha=0.9,
            zorder=4,
        )

        x_min = min(baseline_x.min(), best_x.min())
        x_max = max(baseline_x.max(), best_x.max())
        baseline_x_max = float(baseline_x.max())
        best_x_max = float(best_x.max())
        if record.baseline_tail_start is not None:
            axis.axvspan(
                max(record.baseline_tail_start, x_min),
                baseline_x_max,
                color=INITIAL_EDGE,
                alpha=0.12,
                zorder=0,
            )
            axis.hlines(
                record.baseline_energy,
                max(record.baseline_tail_start, x_min),
                baseline_x_max,
                color=INITIAL_EDGE,
                linewidth=3.2,
                zorder=5,
            )
        if record.best_tail_start is not None:
            axis.axvspan(
                max(record.best_tail_start, x_min),
                best_x_max,
                color=OPT_EDGE,
                alpha=0.12,
                zorder=0,
            )
            axis.hlines(
                record.best_energy,
                max(record.best_tail_start, x_min),
                best_x_max,
                color=OPT_EDGE,
                linewidth=3.4,
                zorder=6,
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
        axis.set_xlim(x_min, x_max)
        _add_param_box(axis, 0.31, 0.97, "Initial", record.baseline_params)
        _add_param_box(axis, 0.66, 0.97, "Optimized", record.best_params)

    axes[0].set_ylabel(r"Energy [Ha]")

    post_eq_handle = (
        Patch(facecolor=INITIAL_EDGE, alpha=0.12, edgecolor="none"),
        Patch(facecolor=OPT_EDGE, alpha=0.12, edgecolor="none"),
    )
    legend_handles = [
        Line2D([0], [0], color=INITIAL_EDGE, linewidth=1.5, marker="o", markersize=6, markerfacecolor=INITIAL_FILL, markeredgewidth=0, label=r"Initial $E(\tau)$"),
        Line2D([0], [0], color=OPT_EDGE, linewidth=1.6, marker="o", markersize=6, markerfacecolor=OPT_FILL, markeredgewidth=0, label=r"Optimized $E(\tau)$"),
        Line2D([0], [0], color=INITIAL_EDGE, linewidth=3.2, label=r"Initial $\langle E_0 \rangle$"),
        Line2D([0], [0], color=OPT_EDGE, linewidth=3.4, label=r"Optimized $\langle E_0 \rangle$"),
        post_eq_handle,
        Patch(facecolor=CHEMICAL_ACCURACY_COLOR, alpha=0.22, label=r"$\pm 1$ kcal/mol"),
        Line2D([0], [0], color=REFERENCE_COLOR, linewidth=2.8, label=records[-1].reference_label),
    ]
    legend_labels = [
        r"Initial $E(\tau)$",
        r"Optimized $E(\tau)$",
        r"Initial $\langle E_0 \rangle$",
        r"Optimized $\langle E_0 \rangle$",
        "Post-equilibration regions",
        r"$\pm 1$ kcal/mol",
        records[-1].reference_label,
    ]
    axes[-1].legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
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
