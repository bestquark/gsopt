from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .common import iso_now, read_json, write_json
from .runtime import RunContext, collect_status


def _append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _freeze_root(context: RunContext) -> Path:
    benchmark_root = context.benchmark_root.resolve()
    if str(context.manifest.get("lane")) in {"vqe", "tn", "afqmc", "dmrg"}:
        return benchmark_root.parent
    return benchmark_root


def _freeze_source(context: RunContext) -> tuple[tempfile.TemporaryDirectory[str], Path, bytes, str]:
    source_bytes = context.source_path.read_bytes()
    source_hash = _sha256_bytes(source_bytes)
    lane = str(context.manifest.get("lane", "benchmark"))
    staging_dir = tempfile.TemporaryDirectory(prefix=f"{lane}_eval_", dir=_freeze_root(context))
    frozen_file = Path(staging_dir.name) / context.source_path.name
    frozen_file.write_bytes(source_bytes)
    return staging_dir, frozen_file, source_bytes, source_hash


def _normalize_result(context: RunContext, result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    if "score" not in normalized:
        metric_key = str(context.manifest.get("objective_metric", "score"))
        if metric_key in normalized:
            normalized["score"] = normalized[metric_key]
    if "score" not in normalized:
        raise RuntimeError("the evaluator must return JSON containing `score` or the manifest objective metric")
    normalized.setdefault("lower_is_better", True)
    return normalized


def _status_from_best(context: RunContext, result: dict[str, Any]) -> tuple[str, int | None, float | None]:
    if context.best_path.exists():
        best = read_json(context.best_path)
        best_score = float(best["score"])
        lower_is_better = bool(best.get("lower_is_better", result.get("lower_is_better", True)))
        current_score = float(result["score"])
        is_better = current_score < best_score if lower_is_better else current_score > best_score
        return ("keep" if is_better else "discard", int(best["iteration"]), best_score)
    return "keep", None, None


def _snapshot_source_file(root: Path, fallback_name: str) -> Path:
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        archived_name = payload.get("archived_source_name")
        if archived_name:
            candidate = root / str(archived_name)
            if candidate.exists():
                return candidate
    candidate = root / fallback_name
    if candidate.exists():
        return candidate
    python_files = sorted(root.glob("*.py"))
    if python_files:
        return python_files[0]
    return candidate


def _previous_snapshot_source(context: RunContext, iteration: int) -> Path | None:
    for idx in range(iteration - 1, -1, -1):
        snapshot_dir = context.local_snapshots_dir / f"iter_{idx:04d}"
        if not snapshot_dir.exists():
            continue
        candidate = _snapshot_source_file(snapshot_dir, context.source_path.name)
        if candidate.exists():
            return candidate
    return None


def _write_diff(previous_file: Path | None, current_file: Path, output_file: Path):
    if previous_file is None:
        output_file.write_text("")
        return
    prev_lines = previous_file.read_text().splitlines(keepends=True)
    curr_lines = current_file.read_text().splitlines(keepends=True)
    diff = difflib.unified_diff(prev_lines, curr_lines, fromfile=str(previous_file), tofile=str(current_file))
    output_file.write_text("".join(diff))


def _evaluator_path(context: RunContext) -> Path:
    copied = context.root_dir / "_user_evaluate.py"
    if copied.exists():
        return copied
    return context.root_dir / str(context.manifest.get("evaluator_file", "evaluate.py"))


def _snapshot_iteration(
    context: RunContext,
    iteration: int,
    description: str,
    source_bytes: bytes,
    source_hash_before: str,
    result: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    error: str | None,
) -> tuple[Path, Path]:
    snapshot_dir = context.local_snapshots_dir / f"iter_{iteration:04d}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    archived_source = snapshot_dir / context.source_path.name
    archived_source.write_bytes(source_bytes)

    live_source_exists = context.source_path.exists()
    live_source_bytes = context.source_path.read_bytes() if live_source_exists else b""
    live_source_hash = _sha256_bytes(live_source_bytes) if live_source_exists else None
    live_source_changed = (not live_source_exists) or (live_source_bytes != source_bytes)

    if result is not None:
        write_json(snapshot_dir / "result.json", result)
    if stdout:
        (snapshot_dir / "stdout.txt").write_text(stdout + "\n")
    if stderr:
        (snapshot_dir / "stderr.txt").write_text(stderr + "\n")
    if error is not None:
        (snapshot_dir / "error.txt").write_text(error + "\n")
    if live_source_changed:
        (snapshot_dir / "live_source_changed.txt").write_text(
            "The live source file changed while this archived iteration was running.\n"
        )

    write_json(
        snapshot_dir / "metadata.json",
        {
            "timestamp": iso_now(),
            "iteration": iteration,
            "description": description,
            "source_file": str(context.source_path),
            "archived_source_name": context.source_path.name,
            "source_sha256_before_run": source_hash_before,
            "live_source_sha256_after_run": live_source_hash,
            "live_source_changed_during_run": live_source_changed,
        },
    )
    previous_file = _previous_snapshot_source(context, iteration)
    _write_diff(previous_file, archived_source, snapshot_dir / "changes.diff")
    return snapshot_dir, archived_source


def _update_best(context: RunContext, iteration: int, result: dict[str, Any], source_snapshot: Path):
    score = float(result["score"])
    lower_is_better = bool(result.get("lower_is_better", True))
    payload = {
        "timestamp": iso_now(),
        "iteration": iteration,
        "score": score,
        "lower_is_better": lower_is_better,
        "source_snapshot": str(source_snapshot),
        "result_path": str(source_snapshot.parent / "result.json"),
    }
    if not context.best_path.exists():
        write_json(context.best_path, payload)
        return
    best = read_json(context.best_path)
    best_score = float(best["score"])
    best_lower = bool(best.get("lower_is_better", lower_is_better))
    is_better = score < best_score if best_lower else score > best_score
    if is_better:
        write_json(context.best_path, payload)


def evaluate_local_context(context: RunContext, description: str, iteration: int | None, extra_args: list[str]) -> int:
    evaluator_path = _evaluator_path(context)
    if not evaluator_path.exists():
        raise RuntimeError(f"missing evaluator file: {evaluator_path}")

    rows = []
    if context.evaluations_log.exists():
        for line in context.evaluations_log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    next_iteration = max((int(row.get("iteration", -1)) for row in rows if isinstance(row.get("iteration"), int)), default=-1) + 1
    actual_iteration = next_iteration if iteration is None else iteration

    cmd = [sys.executable, str(evaluator_path), *extra_args]
    event: dict[str, Any] = {
        "timestamp": iso_now(),
        "type": "evaluation",
        "iteration": actual_iteration,
        "description": description,
        "command": cmd,
    }

    staging_dir, frozen_file, source_bytes, source_hash_before = _freeze_source(context)
    try:
        env = os.environ.copy()
        env["GSOPT_RUN_DIR"] = str(context.root_dir)
        env["GSOPT_ITERATION"] = str(actual_iteration)
        env["GSOPT_SOURCE_FILE"] = str(frozen_file)
        env["GSOPT_SOURCE_NAME"] = context.source_path.name
        env["GSOPT_BENCHMARK_ROOT"] = str(context.benchmark_root)

        proc = subprocess.run(cmd, cwd=context.root_dir, env=env, capture_output=True, text=True, check=False)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        error = None
        result = None

        if proc.returncode != 0:
            error = stderr or stdout or "evaluation failed"
        elif not stdout:
            error = "evaluation produced no stdout"
        else:
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError as exc:
                error = f"evaluation stdout was not valid JSON: {exc}"

        if result is not None:
            result = _normalize_result(context, result)
            if "status" not in result:
                status, previous_best_iteration, previous_best_score = _status_from_best(context, result)
                result["status"] = status
                result["previous_best_iteration"] = previous_best_iteration
                result["previous_best_score"] = previous_best_score
            snapshot_dir, archived_source = _snapshot_iteration(
                context,
                actual_iteration,
                description,
                source_bytes,
                source_hash_before,
                result,
                stdout,
                stderr,
                None,
            )
            if result["status"] == "keep":
                _update_best(context, actual_iteration, result, archived_source)
            event["result"] = result
            event["snapshot_dir"] = str(snapshot_dir)
            _append_jsonl(context.evaluations_log, event)
            collect_status(context, write=True)
            print(json.dumps(result, indent=2))
            return 0

        crash_result = {
            "status": "crash",
            "error": error,
            "lower_is_better": True,
        }
        snapshot_dir, _archived_source = _snapshot_iteration(
            context,
            actual_iteration,
            description,
            source_bytes,
            source_hash_before,
            crash_result,
            stdout,
            stderr,
            error,
        )
        event["result"] = crash_result
        event["error"] = error
        event["snapshot_dir"] = str(snapshot_dir)
        _append_jsonl(context.evaluations_log, event)
        collect_status(context, write=True)
        print(str(error), file=sys.stderr)
        return 1
    finally:
        staging_dir.cleanup()
