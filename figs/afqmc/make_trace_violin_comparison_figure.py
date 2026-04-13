from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.afqmc.model_registry import ACTIVE_SYSTEMS
from figs.afqmc.make_block_trace_figure import (
    CHEMICAL_ACCURACY_COLOR,
    CHEMICAL_ACCURACY_HA,
    INITIAL_EDGE,
    INITIAL_FILL,
    OPT_EDGE,
    OPT_FILL,
    REFERENCE_COLOR,
    TraceRecord,
    _add_param_box,
    configure_style,
    load_record as load_trace_record,
    system_label_tex,
)
from figs.afqmc.make_violin_energy_figure import (
    BEST_EDGE,
    BEST_FILL,
    BEST_MEAN_FILL,
    BEST_SAMPLE_FILL,
    INITIAL_MEAN_FILL,
    INITIAL_SAMPLE_FILL,
    SECONDARY_REFERENCE_COLOR,
    ViolinRecord,
    _add_mean_marker,
    _add_sample_scatter,
    _style_violin,
    load_record as load_violin_record,
)

FIG_DIR = Path(os.environ.get("AUTORESEARCH_AFQMC_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "afqmc_trace_violin_comparison.pdf"
OUTPUT_PNG = FIG_DIR / "afqmc_trace_violin_comparison.png"


def _plot_trace_axis(axis, record: TraceRecord):
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


def _plot_violin_axis(axis, record: ViolinRecord):
    positions = [1.0, 2.0]
    violins = axis.violinplot(
        [record.baseline_samples, record.best_samples],
        positions=positions,
        vert=True,
        widths=0.7,
        showextrema=False,
        showmeans=False,
        showmedians=False,
    )
    _style_violin(violins["bodies"][0], face=INITIAL_FILL, edge=INITIAL_EDGE)
    _style_violin(violins["bodies"][1], face=BEST_FILL, edge=BEST_EDGE)
    _add_sample_scatter(axis, record.baseline_samples, positions[0], face=INITIAL_SAMPLE_FILL, edge=INITIAL_EDGE)
    _add_sample_scatter(axis, record.best_samples, positions[1], face=BEST_SAMPLE_FILL, edge=BEST_EDGE)
    _add_mean_marker(axis, record.baseline_samples, positions[0], face=INITIAL_MEAN_FILL, edge=INITIAL_EDGE)
    _add_mean_marker(axis, record.best_samples, positions[1], face=BEST_MEAN_FILL, edge=BEST_EDGE)

    primary_reference = next((overlay for overlay in record.references if overlay.is_primary), None)
    if primary_reference is not None:
        lower = primary_reference.energy - CHEMICAL_ACCURACY_HA
        upper = primary_reference.energy + CHEMICAL_ACCURACY_HA
        axis.axhspan(lower, upper, color=CHEMICAL_ACCURACY_COLOR, alpha=0.24, zorder=0)
        axis.axhline(primary_reference.energy, color=REFERENCE_COLOR, linestyle="-", linewidth=3.2, zorder=2)
        if primary_reference.stderr is not None and primary_reference.stderr > 0.0:
            axis.axhspan(
                primary_reference.energy - primary_reference.stderr,
                primary_reference.energy + primary_reference.stderr,
                color=REFERENCE_COLOR,
                alpha=0.12,
                zorder=1,
            )

    for overlay in record.references:
        if overlay.is_primary:
            continue
        axis.axhline(
            overlay.energy,
            color=SECONDARY_REFERENCE_COLOR,
            linestyle="--",
            linewidth=2.2,
            zorder=2,
        )

    reference_values = [overlay.energy for overlay in record.references]
    reference_low = min(reference_values) - CHEMICAL_ACCURACY_HA if reference_values else record.best_energy
    reference_high = max(reference_values) + CHEMICAL_ACCURACY_HA if reference_values else record.best_energy
    ymin = min(min(record.baseline_samples), min(record.best_samples), reference_low)
    ymax = max(max(record.baseline_samples), max(record.best_samples), reference_high)
    pad = max(1e-6, 0.08 * (ymax - ymin if ymax > ymin else abs(ymax) * 0.05 + 1e-6))
    axis.set_ylim(ymin - pad, ymax + pad)
    axis.set_xlim(0.4, 2.6)
    axis.set_xticks(positions)
    axis.set_xticklabels([r"Initial", r"Optimized"])
    axis.tick_params(length=6, width=1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    trace_records = [load_trace_record(system) for system in ACTIVE_SYSTEMS]
    violin_records = [load_violin_record(system) for system in ACTIVE_SYSTEMS]
    violin_by_system = {record.system: record for record in violin_records}

    fig, axes = plt.subplots(2, len(trace_records), figsize=(21.5, 10.6), sharey=False)
    if len(trace_records) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for index, trace_record in enumerate(trace_records):
        trace_axis = axes[0, index]
        violin_axis = axes[1, index]
        violin_record = violin_by_system[trace_record.system]

        _plot_trace_axis(trace_axis, trace_record)
        _plot_violin_axis(violin_axis, violin_record)

        trace_axis.set_ylabel("")
        violin_axis.set_ylabel("")

        violin_axis.set_title("")
        trace_axis.tick_params(labelbottom=True)

    mean_handle = (
        Line2D([0], [0], color=INITIAL_EDGE, linewidth=3.2),
        Line2D([0], [0], color=OPT_EDGE, linewidth=3.4),
    )
    post_eq_handle = (
        Patch(facecolor=INITIAL_EDGE, alpha=0.12, edgecolor="none"),
        Patch(facecolor=OPT_EDGE, alpha=0.12, edgecolor="none"),
    )
    legend_handles = [
        Line2D([0], [0], color=INITIAL_EDGE, linewidth=1.5, marker="o", markersize=6, markerfacecolor=INITIAL_FILL, markeredgewidth=0, label=r"Initial $E(\tau)$"),
        Line2D([0], [0], color=OPT_EDGE, linewidth=1.6, marker="o", markersize=6, markerfacecolor=OPT_FILL, markeredgewidth=0, label=r"Optimized $E(\tau)$"),
        mean_handle,
        post_eq_handle,
        Patch(facecolor=CHEMICAL_ACCURACY_COLOR, alpha=0.22, label=r"$\pm 1$ kcal/mol"),
        Line2D([0], [0], color=REFERENCE_COLOR, linewidth=2.8, label=trace_records[-1].reference_label),
    ]
    legend_labels = [
        r"Initial $E(\tau)$",
        r"Optimized $E(\tau)$",
        r"Estimated $\langle E_0 \rangle$",
        "Post-equilibration regions",
        r"$\pm 1$ kcal/mol",
        trace_records[-1].reference_label,
    ]
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(0.865, 0.50),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        alignment="left",
        handlelength=2.4,
        handletextpad=0.7,
        labelspacing=0.7,
    )

    fig.text(0.018, 0.50, r"Energy [Ha]", rotation=90, va="center", ha="center")
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.10, top=0.90, wspace=0.33, hspace=0.20)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "overview_pdf": str(OUTPUT_PDF),
                "overview_png": str(OUTPUT_PNG),
                "runs": {record.system: str(record.run_dir) for record in trace_records},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
