from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes

SNAPSHOT_ROOT = ROOT / "examples" / "vqe" / "snapshots"
FIG_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = FIG_DIR / "vqe_energy_milestones.pdf"
OUTPUT_PNG = FIG_DIR / "vqe_energy_milestones.png"
OUTPUT_TSV = FIG_DIR / "vqe_energy_milestones.tsv"
OUTPUT_MD = FIG_DIR / "vqe_energy_milestones.md"
ORDER = ["bh", "lih", "beh2", "h2o", "n2"]
MOLECULE_NAMES = {
    "bh": "BH",
    "lih": "LiH",
    "beh2": "BeH2",
    "h2o": "H2O",
    "n2": "N2",
}
MOLECULE_LEGEND = {
    "BH": r"$\mathrm{BH}$ -- CAS(2,3)",
    "LiH": r"$\mathrm{LiH}$ -- CAS(2,4)",
    "BeH2": r"$\mathrm{BeH_2}$ -- CAS(4,4)",
    "H2O": r"$\mathrm{H_2O}$ -- CAS(6,4)",
    "N2": r"$\mathrm{N_2}$ -- CAS(6,6)",
}
CHEMICAL_ACCURACY = 1e-3
DELTA_FLOOR = 1e-12
MIN_PREV_ERROR = 1e-8
PRIMARY_RATIO = 2.0
SECONDARY_RATIO = 1.5
NONTRIVIAL_MIN_ITERATION = 4
TRIVIAL_DESCRIPTION_PATTERNS = (
    "switch to uccsd",
    "switch to full uccsd",
    "switch to compressed uccsd",
)
CALLOUT_LAYOUTS = {
    "bh": [((24, 24), 0.0), ((24, 24), 0.0)],
    "lih": [((24, 24), 0.0), ((24, 24), 0.0)],
    "beh2": [((24, 24), 0.0), ((24, 24), 0.0)],
    "h2o": [((24, 24), 0.0), ((24, 24), 0.0)],
    "n2": [((24, 24), 0.0), ((24, 24), 0.0)],
}


@dataclass(frozen=True)
class Improvement:
    stem: str
    molecule: str
    iteration: int
    before_error: float
    after_error: float
    ratio: float
    description: str
    kind: str


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.grid": False,
        }
    )


def molecule_tex(name: str) -> str:
    return MOLECULE_LEGEND.get(name, rf"$\mathrm{{{name}}}$")


def load_rows(stem: str) -> list[dict]:
    path = SNAPSHOT_ROOT / stem / "results.tsv"
    if not path.exists():
        return []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get("iteration")]
    return rows


def running_best_points(rows: list[dict]) -> tuple[list[int], list[float], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    running: list[float] = []
    best = float("inf")
    for row in rows:
        if not row.get("abs_final_error"):
            continue
        delta = max(float(row["abs_final_error"]), DELTA_FLOOR)
        xs.append(int(row["iteration"]))
        ys.append(delta)
        best = min(best, delta)
        running.append(best)
    return xs, ys, running


def improvement_events(stem: str, molecule: str, rows: list[dict]) -> list[Improvement]:
    best = None
    events: list[Improvement] = []
    for row in rows:
        if row.get("status") != "keep" or not row.get("abs_final_error"):
            continue
        error = max(float(row["abs_final_error"]), DELTA_FLOOR)
        if best is None:
            best = error
            continue
        if error < best:
            events.append(
                Improvement(
                    stem=stem,
                    molecule=molecule,
                    iteration=int(row["iteration"]),
                    before_error=best,
                    after_error=error,
                    ratio=best / error,
                    description=row.get("description", "").strip(),
                    kind="",
                )
            )
            best = error
    return events


def select_milestones(events: list[Improvement]) -> list[Improvement]:
    if not events:
        return []

    ranked_nontrivial = [
        event
        for event in events
        if event.iteration >= NONTRIVIAL_MIN_ITERATION
        and event.before_error >= MIN_PREV_ERROR
        and not any(pattern in event.description.lower() for pattern in TRIVIAL_DESCRIPTION_PATTERNS)
    ]
    chosen_pool = ranked_nontrivial or [
        event
        for event in events
        if event.iteration >= NONTRIVIAL_MIN_ITERATION and event.before_error >= MIN_PREV_ERROR
    ]
    if not chosen_pool:
        chosen_pool = [
            event for event in events if event.before_error >= MIN_PREV_ERROR
        ] or events

    chosen_pool = sorted(
        chosen_pool,
        key=lambda event: (event.ratio, event.before_error, event.iteration),
        reverse=True,
    )
    chosen: list[Improvement] = []
    for event in chosen_pool:
        if any(existing.iteration == event.iteration for existing in chosen):
            continue
        chosen.append(
            Improvement(
                stem=event.stem,
                molecule=event.molecule,
                iteration=event.iteration,
                before_error=event.before_error,
                after_error=event.after_error,
                ratio=event.ratio,
                description=event.description,
                kind="Highlighted drop",
            )
        )
        if len(chosen) == 2:
            break

    return sorted(chosen, key=lambda event: event.iteration)


def plot(ax):
    colors = plt.get_cmap("tab10")
    all_deltas: list[float] = []
    handles: list[Line2D] = []
    milestones: list[Improvement] = []

    for idx, stem in enumerate(ORDER):
        rows = load_rows(stem)
        if not rows:
            continue
        molecule = MOLECULE_NAMES[stem]
        xs, ys, running = running_best_points(rows)
        if not xs:
            continue
        color = colors(idx % 10)

        ax.scatter(xs, ys, s=26, color=color, alpha=0.14, zorder=1)
        ax.step(xs, running, where="post", color=color, linewidth=3.2, zorder=2)
        all_deltas.extend(ys)
        handles.append(Line2D([0], [0], color=color, linewidth=3.2, label=molecule_tex(molecule)))
        milestones.extend(select_milestones(improvement_events(stem, molecule, rows)))

    if not all_deltas:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return []

    milestones.sort(key=lambda event: (ORDER.index(event.stem), event.iteration))
    colors_by_stem = {stem: colors(idx % 10) for idx, stem in enumerate(ORDER)}
    seen_per_stem: dict[str, int] = {}
    for marker_id, event in enumerate(milestones, start=1):
        color = colors_by_stem[event.stem]
        local_idx = seen_per_stem.get(event.stem, 0)
        seen_per_stem[event.stem] = local_idx + 1
        offset_points, curve_rad = CALLOUT_LAYOUTS.get(event.stem, [((28, 10), 0.12), ((28, -12), -0.12)])[
            min(local_idx, 1)
        ]
        ax.scatter(
            [event.iteration],
            [event.after_error],
            s=72,
            facecolor="white",
            edgecolor=color,
            linewidth=2.2,
            zorder=4,
        )
        ax.annotate(
            str(marker_id),
            xy=(event.iteration, event.after_error),
            xytext=offset_points,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=15,
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "circle,pad=0.33",
                "fc": "white",
                "ec": color,
                "lw": 2.2,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.8,
                "shrinkA": 4,
                "shrinkB": 5,
                "mutation_scale": 13,
                "connectionstyle": f"arc3,rad={curve_rad}",
            },
            zorder=5,
            clip_on=False,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, max(100, max(int(event.iteration) for event in milestones) if milestones else 1))
    ax.set_ylim(min(all_deltas) / 1.8, max(all_deltas) * 1.8)
    ax.grid(False)
    ax.axhline(CHEMICAL_ACCURACY, color="#222222", linestyle="--", linewidth=2.2, zorder=0)
    finish_axes(ax, xlabel=r"Autoresearch evolutions", ylabel=r"$\Delta E$ [Ha]")
    handles.append(
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.2, label=r"Chemical Accuracy")
    )
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
    )
    legend.get_frame().set_linewidth(1.0)
    return milestones


def write_tables(milestones: list[Improvement]):
    with OUTPUT_TSV.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "id",
                "molecule",
                "kind",
                "iteration",
                "delta_e_before",
                "delta_e_after",
                "improvement_factor",
                "description",
            ]
        )
        for marker_id, event in enumerate(milestones, start=1):
            writer.writerow(
                [
                    marker_id,
                    event.molecule,
                    event.kind,
                    event.iteration,
                    f"{event.before_error:.6e}",
                    f"{event.after_error:.6e}",
                    f"{event.ratio:.3f}",
                    event.description,
                ]
            )

    lines = [
        "| ID | Molecule | Kind | Iter. | $\\Delta E$ before | $\\Delta E$ after | Gain | What changed |",
        "| -: | :-- | :-- | --: | --: | --: | --: | :-- |",
    ]
    for marker_id, event in enumerate(milestones, start=1):
        lines.append(
            "| {id} | {molecule} | {kind} | {iteration} | {before:.3e} | {after:.3e} | {ratio:.1f}x | {desc} |".format(
                id=marker_id,
                molecule=event.molecule,
                kind=event.kind,
                iteration=event.iteration,
                before=event.before_error,
                after=event.after_error,
                ratio=event.ratio,
                desc=event.description.replace("|", "/"),
            )
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    milestones = plot(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    write_tables(milestones)
    print(
        json.dumps(
            {
                "figure_pdf": str(OUTPUT_PDF),
                "figure_png": str(OUTPUT_PNG),
                "milestones_tsv": str(OUTPUT_TSV),
                "milestones_md": str(OUTPUT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
