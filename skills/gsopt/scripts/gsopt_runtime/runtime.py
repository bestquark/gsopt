from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import find_manifest_path, find_repo_root, iso_now, read_json, synthesize_manifest, write_json


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    root_dir: Path
    manifest_path: Path | None
    manifest: dict[str, Any]
    run_meta_path: Path | None
    run_meta: dict[str, Any] | None

    @property
    def is_run(self) -> bool:
        return self.run_meta is not None

    @property
    def benchmark_root(self) -> Path:
        if self.is_run:
            return Path(self.run_meta["benchmark_root"]).resolve()
        return self.root_dir

    @property
    def source_path(self) -> Path:
        return self.root_dir / str(self.manifest["source_file"])

    @property
    def local_snapshots_dir(self) -> Path:
        return self.root_dir / "snapshots"

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "figs"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def evaluations_log(self) -> Path:
        return self.logs_dir / "evaluations.jsonl"

    @property
    def best_path(self) -> Path:
        return self.root_dir / "best.json"

    @property
    def target_iterations(self) -> int:
        if self.run_meta is not None:
            return int(self.run_meta["target_iterations"])
        return int(self.manifest.get("default_iterations", 100))


def locate_context(start: Path) -> RunContext:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    run_meta_path = None
    manifest_path = None
    for candidate in (current, *current.parents):
        if run_meta_path is None and (candidate / "run.json").exists():
            run_meta_path = candidate / "run.json"
        if manifest_path is None:
            manifest_path = find_manifest_path(candidate)
        if run_meta_path is not None or manifest_path is not None:
            break

    repo_root = find_repo_root(current)
    if run_meta_path is not None:
        run_meta = read_json(run_meta_path)
        root_dir = run_meta_path.parent
        manifest_candidate = find_manifest_path(root_dir)
        manifest = read_json(manifest_candidate) if manifest_candidate is not None else synthesize_manifest(root_dir)
        return RunContext(
            repo_root=repo_root,
            root_dir=root_dir,
            manifest_path=manifest_candidate,
            manifest=manifest,
            run_meta_path=run_meta_path,
            run_meta=run_meta,
        )

    if manifest_path is not None:
        return RunContext(
            repo_root=repo_root,
            root_dir=manifest_path.parent,
            manifest_path=manifest_path,
            manifest=read_json(manifest_path),
            run_meta_path=None,
            run_meta=None,
        )

    manifest = synthesize_manifest(current)
    return RunContext(
        repo_root=repo_root,
        root_dir=current,
        manifest_path=None,
        manifest=manifest,
        run_meta_path=None,
        run_meta=None,
    )


def _read_result_metric(context: RunContext, row: dict[str, str]) -> float | None:
    metric_key = str(context.manifest.get("objective_metric", "score"))
    raw = row.get(metric_key, "")
    if raw == "":
        return None
    return float(raw)


def _collect_from_results_tsv(context: RunContext) -> dict[str, Any]:
    benchmark_root = context.local_snapshots_dir / str(context.manifest["benchmark_storage_name"])
    results_path = benchmark_root / "results.tsv"
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
    completed_mutations = max(0, last_iteration or 0)
    return {
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
        "latest_result_path": None
        if last_iteration is None
        else str(benchmark_root / f"iter_{last_iteration:04d}" / "result.json"),
        "figures_dir": str(context.figures_dir),
        "snapshots_dir": str(context.local_snapshots_dir),
        "logs_dir": str(context.logs_dir),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _collect_from_local_log(context: RunContext) -> dict[str, Any]:
    rows = _read_jsonl(context.evaluations_log)
    scored = [row for row in rows if row.get("type") == "evaluation"]
    latest_iteration = None
    best_iteration = None
    best_metric = None
    kept_rows = 0
    for row in scored:
        iteration = row.get("iteration")
        if not isinstance(iteration, int):
            continue
        if latest_iteration is None or iteration > latest_iteration:
            latest_iteration = iteration
        result = row.get("result") or {}
        score = result.get("score")
        status = result.get("status", "keep")
        if score is None or status not in {"keep", "ok", "success"}:
            continue
        kept_rows += 1
        metric = float(score)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_iteration = iteration
    completed_mutations = max(0, latest_iteration or 0)
    best_path = context.best_path if context.best_path.exists() else None
    latest_result_path = None
    if latest_iteration is not None:
        candidate = context.local_snapshots_dir / f"iter_{latest_iteration:04d}" / "result.json"
        if candidate.exists():
            latest_result_path = candidate
    return {
        "timestamp": iso_now(),
        "lane": context.manifest.get("lane", "generic"),
        "benchmark_arg": context.manifest.get("benchmark_arg", "benchmark"),
        "benchmark_value": context.manifest.get("benchmark_value", context.root_dir.name),
        "run_dir": str(context.root_dir),
        "target_iterations": context.target_iterations,
        "evaluation_mode": None if context.run_meta is None else context.run_meta.get("evaluation_mode"),
        "max_parallel": None if context.run_meta is None else context.run_meta.get("max_parallel"),
        "latest_iteration": latest_iteration,
        "completed_mutations": completed_mutations,
        "remaining_mutations": max(context.target_iterations - completed_mutations, 0),
        "done": latest_iteration is not None and latest_iteration >= context.target_iterations,
        "best_iteration": best_iteration,
        "best_metric": best_metric,
        "best_path": None if best_path is None else str(best_path),
        "kept_rows": kept_rows,
        "total_rows": len(scored),
        "results_path": str(context.evaluations_log),
        "latest_result_path": None if latest_result_path is None else str(latest_result_path),
        "figures_dir": str(context.figures_dir),
        "snapshots_dir": str(context.local_snapshots_dir),
        "logs_dir": str(context.logs_dir),
    }


def collect_status(context: RunContext, write: bool = False) -> dict[str, Any]:
    if context.is_run and not context.manifest.get("queue_script") and (
        context.evaluations_log.exists() or (context.root_dir / "_user_evaluate.py").exists()
    ):
        payload = _collect_from_local_log(context)
    else:
        payload = _collect_from_results_tsv(context)
    if write and context.is_run:
        write_json(context.root_dir / "status.json", payload)
    return payload
