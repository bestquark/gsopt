from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes

SNAPSHOT_ROOT = ROOT / "examples" / "vqe" / "snapshots" / "n2_60"
FIG_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = FIG_DIR / "n2_60_optimization_curves.pdf"
OUTPUT_PNG = FIG_DIR / "n2_60_optimization_curves.png"


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 20,
            "axes.grid": False,
        }
    )


def load_results_rows() -> list[dict]:
    path = SNAPSHOT_ROOT / "results.tsv"
    if not path.exists():
        raise SystemExit(f"missing results file: {path}")
    rows = []
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10 or parts[1] == "":
                continue
            rows.append(
                {
                    "iteration": int(parts[0]),
                    "abs_final_error": float(parts[1]),
                    "status": parts[7],
                }
            )
    if not rows:
        raise SystemExit(f"no completed rows found in {path}")
    return rows


def best_kept_iteration(rows: list[dict]) -> int:
    kept = [row for row in rows if row["status"] == "keep"]
    if not kept:
        raise SystemExit("no kept iterations available for N2_60")
    def exact_metric(row: dict) -> tuple[float, int]:
        result = load_result(row["iteration"])
        return (abs(float(result["final_error"])), row["iteration"])

    return min(kept, key=exact_metric)["iteration"]


def load_result(iteration: int) -> dict:
    path = SNAPSHOT_ROOT / f"iter_{iteration:04d}" / "result.json"
    if not path.exists():
        raise SystemExit(f"missing result file: {path}")
    return json.loads(path.read_text())


def snapshot_script(iteration: int) -> Path:
    root = SNAPSHOT_ROOT / f"iter_{iteration:04d}"
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
            archived_name = payload.get("archived_source_name")
            if archived_name:
                path = root / str(archived_name)
                if path.exists():
                    return path
        except json.JSONDecodeError:
            pass
    for name in ("simple_vqe.py",):
        path = root / name
        if path.exists():
            return path
    raise SystemExit(f"missing snapshot source file in {root}")


def load_snapshot_module(iteration: int):
    path = snapshot_script(iteration)
    module_name = f"n2_60_snapshot_iter_{iteration:04d}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not import snapshot module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_history(iteration: int, result: dict) -> tuple[dict, bool]:
    history = result.get("history")
    if history:
        return result, False
    module = load_snapshot_module(iteration)
    wall_seconds = float(result.get("wall_budget_seconds", 60.0))
    problem = module.build_problem(module.MOLECULE_NAME)
    rerun = module.run_config(
        module.DEFAULT_CONFIG,
        problem,
        chemical_accuracy=module.CHEMICAL_ACCURACY,
        wall_time_limit=wall_seconds,
    )
    return rerun, True


def curve_from_result(result: dict) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    running_best = float("inf")
    for step, _energy, error in result["history"]:
        xs.append(int(step))
        running_best = min(running_best, abs(float(error)))
        ys.append(max(running_best, 1e-16))
    return xs, ys


def main():
    rows = load_results_rows()
    baseline, baseline_regenerated = ensure_history(0, load_result(0))
    best_iteration = best_kept_iteration(rows)
    best, best_regenerated = ensure_history(best_iteration, load_result(best_iteration))
    baseline_x, baseline_y = curve_from_result(baseline)
    best_x, best_y = curve_from_result(best)

    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(
        baseline_x,
        baseline_y,
        color="#7a7a7a",
        linewidth=3.0,
        label="Initial setup",
    )
    ax.plot(
        best_x,
        best_y,
        color="#b24a2f",
        linewidth=3.4,
        label="Optimized setup",
    )
    ax.axhline(
        1e-3,
        color="#222222",
        linestyle="--",
        linewidth=2.2,
        label="Chemical Accuracy",
    )

    all_errors = baseline_y + best_y + [1e-3]
    y_min = min(all_errors)
    y_max = max(all_errors)
    ax.set_xlim(1, max(max(baseline_x), max(best_x)))
    ax.set_yscale("log")
    ax.set_ylim(max(1e-16, y_min / 1.8), y_max * 1.8)
    finish_axes(ax, xlabel=r"VQE Iterations", ylabel=r"$\Delta E$ [Ha]")
    ax.grid(False)

    legend = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#444444")
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "overview_pdf": str(OUTPUT_PDF),
                "overview_png": str(OUTPUT_PNG),
                "best_iteration": best_iteration,
                "baseline_history_regenerated": baseline_regenerated,
                "best_history_regenerated": best_regenerated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
