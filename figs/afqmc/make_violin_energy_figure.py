from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style
from examples.afqmc.model_registry import ACTIVE_SYSTEMS
from examples.afqmc.reference_energies import reference_energy, reference_sources, reference_stderr

LANE_DIR = ROOT / "examples" / "afqmc"
FIG_DIR = Path(os.environ.get("AUTORESEARCH_AFQMC_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "afqmc_violin_energy_comparison.pdf"
OUTPUT_PNG = FIG_DIR / "afqmc_violin_energy_comparison.png"
RUN_SELECTION = os.environ.get("AUTORESEARCH_AFQMC_RUN_SELECTION", "completed").strip().lower()
INITIAL_FILL = "#d8a5a5"
INITIAL_EDGE = "#b14f4f"
BEST_FILL = "#b9d97c"
BEST_EDGE = "#76B900"
INITIAL_SAMPLE_FILL = "#d7b9b9"
BEST_SAMPLE_FILL = "#d4e5ad"
INITIAL_MEAN_FILL = "#fff4f4"
BEST_MEAN_FILL = "#f8ffe9"
REFERENCE_COLOR = "#222222"
SECONDARY_REFERENCE_COLOR = "#6b6b6b"
CHEMICAL_ACCURACY_COLOR = "#8d8d8d"
CHEMICAL_ACCURACY_HA = 1.0 / 627.509474


@dataclass(frozen=True)
class ReferenceOverlay:
    method_key: str
    label: str
    energy: float
    stderr: float | None
    is_primary: bool


@dataclass(frozen=True)
class ViolinRecord:
    system: str
    run_dir: Path
    baseline_iteration: int
    best_iteration: int
    references: tuple[ReferenceOverlay, ...]
    baseline_energy: float
    best_energy: float
    baseline_samples: list[float]
    best_samples: list[float]


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
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


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
    final_energy = result.get("final_energy")
    if isinstance(final_energy, (int, float)):
        return float(final_energy)
    return None


def _extract_samples(result: dict) -> list[float]:
    for key in ("block_averaged_energies", "mixed_estimator_energies"):
        values = result.get(key)
        if isinstance(values, list) and values:
            return [float(value) for value in values]

    final_energy = float(result["final_energy"])
    epsilon = max(1e-9, abs(final_energy) * 1e-9)
    return [final_energy - epsilon, final_energy, final_energy + epsilon]


def _reference_label(source: dict) -> str:
    method = str(source.get("reference_method", "")).strip()
    if method.lower() in {"ph_afqmc", "ph-afqmc"}:
        return r"ph-AFQMC"
    return method or "Reference"


def _reference_overlays(system: str) -> tuple[ReferenceOverlay, ...]:
    sources = reference_sources(system)
    primary_energy = reference_energy(system)
    primary_stderr = reference_stderr(system)
    overlays: list[ReferenceOverlay] = []
    primary_key: str | None = None
    for key, source in sources.items():
        energy = source.get("reference_energy")
        if energy is None:
            continue
        energy_value = float(energy)
        stderr = source.get("stderr")
        stderr_value = None if stderr is None else float(stderr)
        is_primary = False
        if primary_energy is not None and abs(energy_value - primary_energy) <= 1e-12:
            if primary_stderr is None or stderr_value == primary_stderr:
                is_primary = True
                primary_key = key
        overlays.append(
            ReferenceOverlay(
                method_key=key,
                label=_reference_label(source),
                energy=energy_value,
                stderr=stderr_value,
                is_primary=is_primary,
            )
        )
    if overlays and primary_key is None:
        overlays[0] = ReferenceOverlay(
            method_key=overlays[0].method_key,
            label=overlays[0].label,
            energy=overlays[0].energy,
            stderr=overlays[0].stderr,
            is_primary=True,
        )
    overlays.sort(key=lambda overlay: (not overlay.is_primary, overlay.method_key))
    return tuple(overlays)


def load_record(system: str) -> ViolinRecord:
    run_dir = _latest_run_dir(system)
    baseline_iteration = 0
    baseline_result = _result_from_snapshot(run_dir / "snapshots" / "iter_0000")
    if baseline_result is None:
        raise FileNotFoundError(f"missing baseline result.json for {run_dir}")
    best_iteration = baseline_iteration
    best_result = baseline_result
    best_score = _result_score(baseline_result)
    if best_score is None:
        raise ValueError(f"baseline result for {run_dir} does not contain a valid score")
    for snapshot_dir in sorted((run_dir / "snapshots").glob("iter_*")):
        result = _result_from_snapshot(snapshot_dir)
        if result is None:
            continue
        try:
            iteration = int(snapshot_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        candidate_score = _result_score(result)
        if candidate_score is None:
            continue
        if candidate_score < best_score:
            best_iteration = iteration
            best_result = result
            best_score = candidate_score
    return ViolinRecord(
        system=system,
        run_dir=run_dir,
        baseline_iteration=baseline_iteration,
        best_iteration=best_iteration,
        references=_reference_overlays(system),
        baseline_energy=float(baseline_result["final_energy"]),
        best_energy=float(best_result["final_energy"]),
        baseline_samples=_extract_samples(baseline_result),
        best_samples=_extract_samples(best_result),
    )


def _style_violin(body, *, face: str, edge: str):
    body.set_facecolor(face)
    body.set_edgecolor(edge)
    body.set_alpha(0.85)
    body.set_linewidth(2.0)


def _scatter_offsets(count: int, *, span: float = 0.10) -> list[float]:
    if count <= 1:
        return [0.0]
    if count == 2:
        return [-0.5 * span, 0.5 * span]
    return [(-0.5 * span) + span * index / (count - 1) for index in range(count)]


def _add_sample_scatter(ax, samples: list[float], center: float, *, face: str, edge: str):
    offsets = _scatter_offsets(len(samples))
    ax.scatter(
        [center + offset for offset in offsets],
        samples,
        s=60,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
        alpha=0.82,
        zorder=4,
    )


def _add_mean_marker(ax, samples: list[float], center: float, *, face: str, edge: str):
    ax.scatter(
        [center],
        [fmean(samples)],
        s=125,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.6,
        alpha=1.0,
        zorder=5,
    )


def _set_energy_ticks(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def plot_record(ax, record: ViolinRecord):
    positions = [1.0, 2.0]
    violins = ax.violinplot(
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
    _add_sample_scatter(ax, record.baseline_samples, positions[0], face=INITIAL_SAMPLE_FILL, edge=INITIAL_EDGE)
    _add_sample_scatter(ax, record.best_samples, positions[1], face=BEST_SAMPLE_FILL, edge=BEST_EDGE)
    _add_mean_marker(ax, record.baseline_samples, positions[0], face=INITIAL_MEAN_FILL, edge=INITIAL_EDGE)
    _add_mean_marker(ax, record.best_samples, positions[1], face=BEST_MEAN_FILL, edge=BEST_EDGE)

    primary_reference = next((overlay for overlay in record.references if overlay.is_primary), None)
    if primary_reference is not None:
        lower = primary_reference.energy - CHEMICAL_ACCURACY_HA
        upper = primary_reference.energy + CHEMICAL_ACCURACY_HA
        ax.axhspan(
            lower,
            upper,
            color=CHEMICAL_ACCURACY_COLOR,
            alpha=0.24,
            zorder=0,
        )
        ax.axhline(
            primary_reference.energy,
            color=REFERENCE_COLOR,
            linestyle="-",
            linewidth=3.2,
            zorder=2,
        )
        if primary_reference.stderr is not None and primary_reference.stderr > 0.0:
            ax.axhspan(
                primary_reference.energy - primary_reference.stderr,
                primary_reference.energy + primary_reference.stderr,
                color=REFERENCE_COLOR,
                alpha=0.12,
                zorder=1,
            )

    for overlay in record.references:
        if overlay.is_primary:
            continue
        ax.axhline(
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
    ax.set_ylim(ymin - pad, ymax + pad)
    _set_energy_ticks(ax)
    ax.set_xlim(0.4, 2.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([r"Initial", r"Optimized"])
    ax.set_title(system_label_tex(record.system), pad=14)
    ax.tick_params(length=6, width=1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    records = [load_record(system) for system in ACTIVE_SYSTEMS]

    fig, axes = plt.subplots(1, len(records), figsize=(22.0, 6.8))
    if len(records) == 1:
        axes = [axes]

    for index, (ax, record) in enumerate(zip(axes, records, strict=True)):
        plot_record(ax, record)
        ax.set_xlabel("")
        if index == 0:
            ax.set_ylabel(r"Energy [Ha]")
        else:
            ax.set_ylabel("")

    reference_labels: dict[str, ReferenceOverlay] = {}
    for record in records:
        for overlay in record.references:
            reference_labels.setdefault(overlay.method_key, overlay)
    legend_handles = []
    primary_overlay = next((overlay for overlay in reference_labels.values() if overlay.is_primary), None)
    if primary_overlay is not None:
        primary_label = primary_overlay.label
        if primary_overlay.stderr is not None and primary_overlay.stderr > 0.0:
            primary_label = rf"{primary_label} $\pm 1\sigma$"
            legend_handles.append(
                Patch(
                    facecolor=REFERENCE_COLOR,
                    edgecolor=REFERENCE_COLOR,
                    alpha=0.12,
                    label=primary_label,
                )
            )
        else:
            legend_handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linewidth=3.2, label=primary_label))
    else:
        legend_handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linewidth=3.2, label="Primary reference"))
    secondary_overlays = [overlay for overlay in reference_labels.values() if not overlay.is_primary]
    for overlay in sorted(secondary_overlays, key=lambda item: item.method_key):
        legend_handles.append(
            Line2D([0], [0], color=SECONDARY_REFERENCE_COLOR, linewidth=2.2, linestyle="--", label=overlay.label)
        )
    legend_handles.append(
        Patch(
            facecolor=CHEMICAL_ACCURACY_COLOR,
            edgecolor=CHEMICAL_ACCURACY_COLOR,
            alpha=0.24,
            label=r"$\pm 1$ kcal/mol",
        )
    )
    axes[-1].legend(
        handles=legend_handles,
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1.08, 0.50),
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.17, top=0.88, wspace=0.62)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    payload = {
        "overview_pdf": str(OUTPUT_PDF),
        "overview_png": str(OUTPUT_PNG),
        "runs": {record.system: str(record.run_dir) for record in records},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
