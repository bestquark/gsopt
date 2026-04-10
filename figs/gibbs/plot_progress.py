from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_records(run_dir: Path) -> list[dict]:
    path = run_dir / "iterations.jsonl"
    records = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def extract_metric(record: dict, metric: str) -> float:
    if metric in record:
        return float(record[metric])
    result = record.get("result", {})
    if metric in result:
        return float(result[metric])
    if metric == "normalized_error":
        source = result or record
        if {"final_error", "hf_energy", "target_energy"} <= source.keys():
            denom = max(abs(float(source["hf_energy"]) - float(source["target_energy"])), 1e-3)
            return float(source["final_error"]) / denom
    aggregate = record.get("aggregate", {})
    if metric in aggregate:
        return float(aggregate[metric])
    raise KeyError(f"metric {metric!r} not found in record")


def pretty_ansatz(name: str | None) -> str:
    mapping = {
        "uccsd_jw": "UCCSD-JW",
        "hea_ry_ring": "Ry ring",
        "hea_ryrz_ring": "Ry-Rz ring",
    }
    return mapping.get(name or "", (name or "").replace("_", " "))


def pretty_optimizer(name: str | None) -> str:
    mapping = {"adam": "Adam", "bfgs": "BFGS", "cobyla": "COBYLA"}
    return mapping.get(name or "", name or "")


def describe_config(config: dict) -> str:
    if "ansatz" in config:
        parts = [
            pretty_ansatz(config.get("ansatz")),
            f"{config.get('layers')}L",
            pretty_optimizer(config.get("optimizer")),
        ]
        if config.get("optimizer") == "adam" and config.get("lr") is not None:
            parts.append(f"lr {config.get('lr')}")
        if config.get("max_steps") is not None:
            parts.append(f"{config.get('max_steps')} steps")
        return ", ".join(parts)
    if "bond_dims" in config:
        bond = ",".join(str(x) for x in config.get("bond_dims", []))
        return f"bond [{bond}], sweeps {config.get('max_sweeps')}"
    if config.get("method") == "low_rank":
        return f"low-rank k={config.get('rank')}"
    if config.get("method") == "product":
        ansatz = str(config.get("ansatz", "")).replace("_", "-")
        return f"{ansatz}, {config.get('optimizer')}, {config.get('maxiter')}"
    return config.get("name", "candidate")


def infer_yscale(values: list[float], requested: str) -> str:
    if requested != "auto":
        return requested
    positive = [v for v in values if v > 0.0 and math.isfinite(v)]
    if len(positive) < len(values):
        return "linear"
    if not positive:
        return "linear"
    spread = max(positive) / max(min(positive), 1e-30)
    return "log" if spread >= 100.0 else "linear"


def main():
    parser = argparse.ArgumentParser(description="Render a Karpathy-style autoresearch progress figure.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metric")
    parser.add_argument("--ylabel")
    parser.add_argument("--title")
    parser.add_argument("--yscale", choices=["auto", "linear", "log"], default="auto")
    parser.add_argument("--out")
    parser.add_argument("--summary-json")
    args = parser.parse_args()

    run_dir = args.run_dir
    manifest = load_manifest(run_dir)
    records = load_records(run_dir)
    if not records:
        raise SystemExit(f"no iteration records found in {run_dir}")

    objective = manifest.get("objective", {})
    metric = args.metric or objective.get("metric") or "score"
    ylabel = args.ylabel or objective.get("label") or metric

    points = []
    best_value = None
    best_iteration = None
    kept_points = []
    discarded_points = []
    running_x = []
    running_y = []

    for record in records:
        iteration = int(record["iteration"])
        status = record.get("status", "completed")
        try:
            value = extract_metric(record, metric)
        except KeyError:
            continue
        if not math.isfinite(value):
            continue

        points.append((iteration, value))
        if status != "completed":
            discarded_points.append((iteration, value))
            continue

        if best_value is None or value < best_value:
            best_value = value
            best_iteration = iteration
            kept_points.append((iteration, value, describe_config(record.get("config", {}))))
        else:
            discarded_points.append((iteration, value))

        running_x.append(iteration)
        running_y.append(best_value)

    if best_value is None:
        raise SystemExit(f"no finite {metric} values found in {run_dir}")

    values = [value for _, value in points]
    yscale = infer_yscale(values, args.yscale)

    apply_style()
    fig, ax = plt.subplots(figsize=(11.6, 6.4))

    if discarded_points:
        ax.scatter(
            [x for x, _ in discarded_points],
            [y for _, y in discarded_points],
            s=11,
            c="#cfcfcf",
            alpha=0.45,
            label="Discarded",
            zorder=2,
        )
    ax.scatter(
        [x for x, _, _ in kept_points],
        [y for _, y, _ in kept_points],
        s=48,
        c="#3ecf78",
        edgecolors="#1f6b44",
        linewidths=0.9,
        label="Kept",
        zorder=4,
    )
    ax.step(
        running_x,
        running_y,
        where="post",
        color="#69d59b",
        linewidth=1.9,
        label="Running best",
        zorder=3,
    )

    for idx, (x, y, label) in enumerate(kept_points):
        if idx == 0:
            label_text = "baseline"
        else:
            label_text = label
        ax.annotate(
            label_text,
            xy=(x, y),
            xytext=(5, 6),
            textcoords="offset points",
            rotation=30,
            fontsize=9,
            color="#2c9d63",
            alpha=0.95,
        )

    finish_axes(ax, xlabel="Experiment #", ylabel=f"{ylabel} (lower is better)")
    ax.set_yscale(yscale)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.16)

    title = args.title or f"{manifest.get('problem_label', 'Autoresearch')} Progress: {len(points)} Experiments, {len(kept_points)} Kept Improvements"
    ax.set_title(title)

    lane = manifest.get("lane", "misc")
    out_path = Path(args.out) if args.out else ROOT / "figs" / lane / f"{run_dir.name}_progress.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "problem_label": manifest.get("problem_label"),
        "metric": metric,
        "ylabel": ylabel,
        "total_points": len(points),
        "kept_improvements": len(kept_points),
        "best_iteration": best_iteration,
        "best_value": best_value,
        "best_label": kept_points[-1][2] if kept_points else None,
        "output": str(out_path),
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
