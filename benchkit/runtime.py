from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import find_manifest_path, iso_now, read_json, write_json


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    root_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    run_meta_path: Path | None
    run_meta: dict[str, Any] | None

    @property
    def is_run(self) -> bool:
        return self.run_meta is not None

    @property
    def benchmark_root(self) -> Path:
        return self.root_dir / "snapshots" / str(self.manifest["benchmark_storage_name"])

    @property
    def results_path(self) -> Path:
        return self.benchmark_root / "results.tsv"

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "figs"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def target_iterations(self) -> int:
        if self.run_meta is not None:
            return int(self.run_meta["target_iterations"])
        return int(self.manifest["default_iterations"])


def locate_context(start: Path) -> RunContext:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    manifest_path = None
    run_meta_path = None
    for candidate in (current, *current.parents):
        if manifest_path is None:
            manifest_path = find_manifest_path(candidate)
        if run_meta_path is None and (candidate / "run.json").exists():
            run_meta_path = candidate / "run.json"
        if manifest_path is not None and (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            repo_root = candidate
            break
    else:
        raise FileNotFoundError(f"could not locate gsopt context from {start}")
    manifest = read_json(manifest_path)
    run_meta = read_json(run_meta_path) if run_meta_path is not None else None
    root_dir = run_meta_path.parent if run_meta_path is not None else manifest_path.parent
    return RunContext(
        repo_root=repo_root,
        root_dir=root_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        run_meta_path=run_meta_path,
        run_meta=run_meta,
    )


def _read_result_metric(context: RunContext, row: dict[str, str]) -> float | None:
    metric_key = str(context.manifest["objective_metric"])
    raw = row.get(metric_key, "")
    if raw == "":
        return None
    return float(raw)


def _latest_result_path(context: RunContext, iteration: int | None) -> Path | None:
    if iteration is None or iteration < 0:
        return None
    result_path = context.benchmark_root / f"iter_{iteration:04d}" / "result.json"
    return result_path if result_path.exists() else None


def collect_status(context: RunContext, write: bool = False) -> dict[str, Any]:
    results_path = context.results_path
    last_iteration = None
    best_iteration = None
    best_metric = None
    kept_rows = 0
    total_rows = 0
    if results_path.exists():
        with results_path.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                iteration_text = row.get("iteration", "")
                if not iteration_text:
                    continue
                iteration = int(iteration_text)
                total_rows += 1
                if last_iteration is None or iteration > last_iteration:
                    last_iteration = iteration
                if row.get("status") != "keep":
                    continue
                metric = _read_result_metric(context, row)
                if metric is None:
                    continue
                kept_rows += 1
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_iteration = iteration

    latest_result_path = _latest_result_path(context, last_iteration)
    completed_mutations = max(0, last_iteration or 0)
    payload = {
        "timestamp": iso_now(),
        "lane": context.manifest["lane"],
        "benchmark_arg": context.manifest["benchmark_arg"],
        "benchmark_value": context.manifest["benchmark_value"],
        "run_dir": str(context.root_dir),
        "target_iterations": context.target_iterations,
        "evaluation_mode": None if context.run_meta is None else context.run_meta.get("evaluation_mode"),
        "max_parallel": None if context.run_meta is None else context.run_meta.get("max_parallel"),
        "latest_iteration": last_iteration,
        "completed_mutations": completed_mutations,
        "remaining_mutations": max(context.target_iterations - completed_mutations, 0),
        "done": last_iteration is not None and last_iteration >= context.target_iterations,
        "best_iteration": best_iteration,
        "best_metric": best_metric,
        "kept_rows": kept_rows,
        "total_rows": total_rows,
        "results_path": str(results_path),
        "latest_result_path": None if latest_result_path is None else str(latest_result_path),
        "figures_dir": str(context.figures_dir),
        "snapshots_dir": str(context.root_dir / "snapshots"),
        "logs_dir": str(context.logs_dir),
    }
    if write and context.is_run:
        write_json(context.root_dir / "status.json", payload)
    return payload


def append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{payload}\n")
