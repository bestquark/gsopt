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
from examples.vqe.reference_energies import reference_energy

LANE_DIR = ROOT / "examples" / "vqe"
CONFIG_DIR = LANE_DIR / "configs"
RUN_ROOT = Path(os.environ.get("AUTORESEARCH_VQE_RUN_ROOT", LANE_DIR / "runs"))
SNAPSHOT_ROOT = Path(
    os.environ.get("AUTORESEARCH_VQE_SNAPSHOT_ROOT", LANE_DIR / "snapshots")
)
OPTUNA_ROOT = Path(os.environ.get("AUTORESEARCH_VQE_OPTUNA_ROOT", LANE_DIR / "optuna"))
FIG_DIR = Path(os.environ.get("AUTORESEARCH_VQE_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "vqe_energy_overview.pdf"
OUTPUT_PNG = FIG_DIR / "vqe_energy_overview.png"
COMPARE_PDF = FIG_DIR / "vqe_energy_overview_with_optuna.pdf"
COMPARE_PNG = FIG_DIR / "vqe_energy_overview_with_optuna.png"
ORDER = ["bh", "lih", "beh2", "h2o", "n2"]
DELTA_FLOOR = 1e-12
CHEMICAL_ACCURACY = 1e-3
OPTUNA_LINESTYLE = (0, (8, 3, 1.8, 3))


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
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "axes.grid": False,
        }
    )


def molecule_tex(name: str) -> str:
    mapping = {
        "N2": r"$\mathrm{N_2}$",
        "BH": r"$\mathrm{BH}$",
        "LiH": r"$\mathrm{LiH}$",
        "BeH2": r"$\mathrm{BeH_2}$",
        "H2O": r"$\mathrm{H_2O}$",
    }
    return mapping.get(name, rf"$\mathrm{{{name}}}$")


def load_config_rows() -> list[dict]:
    rows = []
    for stem in ORDER:
        path = CONFIG_DIR / f"{stem}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


def load_records(run_name: str) -> list[dict]:
    run_dir = RUN_ROOT / run_name
    path = run_dir / "iterations.jsonl"
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_snapshot_records(stem: str) -> list[dict]:
    snapshot_dir = SNAPSHOT_ROOT / stem
    if not snapshot_dir.exists():
        return []
    rows = []
    for child in sorted(snapshot_dir.iterdir()):
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
        rows.append(
            {
                "iteration": iteration,
                "final_energy": result["final_energy"],
                "result": result,
            }
        )
    return rows


def load_optuna_records(stem: str) -> list[dict]:
    root = locate_optuna_root(stem)
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
        rows.append(
            {
                "iteration": iteration,
                "final_energy": result["final_energy"],
                "result": result,
            }
        )
    return rows


def locate_optuna_root(stem: str) -> Path | None:
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


def comparison_autoresearch_records(config: dict) -> list[dict]:
    stem = config["molecule"].lower().replace("+", "_plus")
    records = load_snapshot_records(stem) or load_records(config["run_name"])
    return [record for record in records if int(record["iteration"]) > 0]


def vqe_delta(result: dict, molecule: str) -> float | None:
    if result.get("abs_final_error") is not None:
        return max(abs(float(result["abs_final_error"])), DELTA_FLOOR)
    if result.get("final_error") is not None:
        return max(abs(float(result["final_error"])), DELTA_FLOOR)
    target = result.get("target_energy")
    if target is None:
        target = reference_energy(molecule)
    if target is None:
        return None
    return max(abs(float(result["final_energy"]) - float(target)), DELTA_FLOOR)


def make_combined_plot(ax, configs: list[dict]):
    colors = plt.get_cmap("tab10")
    min_iteration = 0
    max_iteration = 0
    all_deltas: list[float] = []
    handles: list[Line2D] = []

    for idx, config in enumerate(configs):
        stem = config["molecule"].lower().replace("+", "_plus")
        records = load_snapshot_records(stem)
        if not records:
            records = load_records(config["run_name"])
        if not records:
            continue

        result0 = records[0]["result"]
        molecule = str(result0.get("molecule", config["molecule"]))
        cas = tuple(result0["cas"])
        color = colors(idx % 10)

        points = [
            (int(record["iteration"]), vqe_delta(record["result"], molecule))
            for record in records
        ]
        points = [(iteration, delta) for iteration, delta in points if delta is not None]
        if not points:
            continue
        xs = [iteration for iteration, _delta in points]
        ys = [delta for _iteration, delta in points]
        running = []
        best = float("inf")
        for value in ys:
            best = min(best, value)
            running.append(best)

        min_iteration = min(min_iteration, min(xs))
        max_iteration = max(max_iteration, max(xs))
        all_deltas.extend(ys)

        ax.scatter(xs, ys, s=34, color=color, alpha=0.22, zorder=1)
        ax.step(xs, running, where="post", color=color, linewidth=3.4, zorder=2)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=3.4,
                label=rf"{molecule_tex(molecule)} -- $\mathrm{{CAS}}({cas[0]},{cas[1]})$",
            )
        )

    if not all_deltas:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return

    ax.set_yscale("log")
    right = max_iteration if max_iteration > min_iteration else min_iteration + 1
    ax.set_xlim(min_iteration, right)
    ax.set_ylim(min(all_deltas) / 1.8, max(all_deltas) * 1.8)
    ax.grid(False)
    ax.axhline(CHEMICAL_ACCURACY, color="#222222", linestyle="--", linewidth=2.2, zorder=0)
    finish_axes(ax, xlabel=r"Iterations", ylabel=r"$\Delta E$ [Ha]")

    handles.append(
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=2.2,
            label=r"Chemical Accuracy",
        )
    )
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    legend.get_frame().set_linewidth(1.0)


def has_optuna_data(configs: list[dict]) -> bool:
    for config in configs:
        stem = config["molecule"].lower().replace("+", "_plus")
        autoresearch_records = comparison_autoresearch_records(config)
        optuna_records = load_optuna_records(stem)
        if autoresearch_records and optuna_records:
            return True
    return False


def make_comparison_plot(ax, configs: list[dict]):
    colors = plt.get_cmap("tab10")
    min_iteration = 0
    max_iteration = 0
    all_deltas: list[float] = []
    molecule_handles: list[Line2D] = []

    for idx, config in enumerate(configs):
        stem = config["molecule"].lower().replace("+", "_plus")
        autoresearch_records = comparison_autoresearch_records(config)
        optuna_records = load_optuna_records(stem)
        if not autoresearch_records or not optuna_records:
            continue

        base_record = (autoresearch_records or optuna_records)[0]["result"]
        molecule = str(base_record.get("molecule", config["molecule"]))
        cas = tuple(base_record["cas"])
        color = colors(idx % 10)

        if autoresearch_records:
            points = [
                (index, vqe_delta(record["result"], molecule))
                for index, record in enumerate(autoresearch_records, start=1)
            ]
            points = [(iteration, delta) for iteration, delta in points if delta is not None]
            xs = [iteration for iteration, _delta in points]
            ys = [delta for _iteration, delta in points]
            running = []
            best = float("inf")
            for value in ys:
                best = min(best, value)
                running.append(best)
            if ys:
                min_iteration = min(min_iteration, min(xs))
                max_iteration = max(max_iteration, max(xs))
                all_deltas.extend(ys)
                ax.step(xs, running, where="post", color=color, linewidth=3.0, zorder=2)

        if optuna_records:
            points = [
                (index, vqe_delta(record["result"], molecule))
                for index, record in enumerate(optuna_records, start=1)
            ]
            points = [(iteration, delta) for iteration, delta in points if delta is not None]
            xs = [iteration for iteration, _delta in points]
            ys = [delta for _iteration, delta in points]
            if autoresearch_records and ys:
                # Evaluation 1 is the shared seeded start for both methods.
                first_autoresearch = vqe_delta(autoresearch_records[0]["result"], molecule)
                if first_autoresearch is not None:
                    ys[0] = first_autoresearch
            running = []
            best = float("inf")
            for value in ys:
                best = min(best, value)
                running.append(best)
            if ys:
                min_iteration = min(min_iteration, min(xs))
                max_iteration = max(max_iteration, max(xs))
                all_deltas.extend(ys)
                ax.step(xs, running, where="post", color=color, linewidth=2.6, linestyle=OPTUNA_LINESTYLE, zorder=4)

        molecule_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=3.0,
                label=rf"{molecule_tex(molecule)} -- $\mathrm{{CAS}}({cas[0]},{cas[1]})$",
            )
        )

    if not all_deltas:
        ax.text(0.5, 0.5, "no run data", transform=ax.transAxes, ha="center", va="center")
        return

    ax.set_yscale("log")
    right = max_iteration if max_iteration > min_iteration else min_iteration + 1
    ax.set_xlim(min_iteration, right)
    ax.set_ylim(min(all_deltas) / 1.8, max(all_deltas) * 1.8)
    ax.axhline(CHEMICAL_ACCURACY, color="#222222", linestyle="--", linewidth=2.2, zorder=0)
    finish_axes(ax, xlabel=r"Evaluator Calls", ylabel=r"$\Delta E$ [Ha]")

    style_handles = [
        Line2D([0], [0], color="#111111", linewidth=3.0, label="Autoresearch"),
        Line2D([0], [0], color="#111111", linewidth=2.6, linestyle=OPTUNA_LINESTYLE, label="Optuna BO (TPE)"),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.2, label=r"Chemical Accuracy"),
    ]
    legend = ax.legend(
        handles=molecule_handles + style_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    legend.get_frame().set_linewidth(1.0)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_config_rows()
    configure_style()
    fig, ax = plt.subplots(figsize=(13, 8.5))
    make_combined_plot(ax, rows)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)

    payload = {"overview_pdf": str(OUTPUT_PDF), "overview_png": str(OUTPUT_PNG)}
    if has_optuna_data(rows):
        fig, ax = plt.subplots(figsize=(13, 8.5))
        make_comparison_plot(ax, rows)
        fig.tight_layout()
        fig.savefig(COMPARE_PDF, bbox_inches="tight")
        fig.savefig(COMPARE_PNG, dpi=240, bbox_inches="tight")
        plt.close(fig)
        payload["comparison_pdf"] = str(COMPARE_PDF)
        payload["comparison_png"] = str(COMPARE_PNG)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
