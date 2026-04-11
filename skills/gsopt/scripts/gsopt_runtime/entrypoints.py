from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import iso_now, read_json
from .local_eval import evaluate_local_context
from .runtime import RunContext, collect_status, locate_context


def _context_from_argv0() -> RunContext:
    return locate_context(Path(sys.argv[0]).resolve())


def _append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _subprocess_json(cmd: list[str], env: dict[str, str], cwd: Path) -> dict:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command stdout was not valid JSON: {exc}") from exc


def evaluate_main() -> int:
    context = _context_from_argv0()
    default_max_parallel = 1
    if context.run_meta is not None:
        default_max_parallel = int(context.run_meta.get("max_parallel", 1))
    parser = argparse.ArgumentParser(
        description="Run one tracked scored evaluation inside a GSOpt benchmark or run directory."
    )
    parser.add_argument("--description", default="", help="Short mutation summary.")
    parser.add_argument("--wall-seconds", type=float, default=float(context.manifest.get("default_wall_seconds", 20.0)))
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--max-parallel", type=int, default=default_max_parallel)
    args, remainder = parser.parse_known_args()

    if not context.manifest.get("queue_script"):
        return evaluate_local_context(context, args.description, args.iteration, remainder)

    manifest = context.manifest
    queue_script = manifest.get("queue_script")
    env = os.environ.copy()
    snapshot_env = manifest.get("snapshot_env")
    if snapshot_env:
        env[str(snapshot_env)] = str(context.root_dir / "snapshots")

    cmd = [
        sys.executable,
        str(context.repo_root / queue_script),
        "--script",
        str(context.root_dir / manifest["source_file"]),
        f"--{manifest['benchmark_arg']}",
        str(manifest["benchmark_value"]),
        "--wall-seconds",
        str(args.wall_seconds),
        "--max-parallel",
        str(args.max_parallel),
        "--description",
        args.description,
    ]
    if args.iteration is not None:
        cmd.extend(["--iteration", str(args.iteration)])

    event = {
        "timestamp": iso_now(),
        "type": "evaluation",
        "command": cmd,
        "description": args.description,
    }
    try:
        result = _subprocess_json(cmd, env=env, cwd=context.repo_root)
        event["result"] = result
        _append_jsonl(context.logs_dir / "evaluations.jsonl", event)
        collect_status(context, write=True)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        event["error"] = str(exc)
        _append_jsonl(context.logs_dir / "evaluations.jsonl", event)
        collect_status(context, write=True)
        print(str(exc), file=sys.stderr)
        return 1


def restore_main() -> int:
    context = _context_from_argv0()
    if not context.manifest.get("restore_script") and context.best_path.exists():
        best = read_json(context.best_path)
        snapshot = Path(best["source_snapshot"])
        if not snapshot.exists():
            print(f"missing best snapshot {snapshot}", file=sys.stderr)
            return 1
        shutil.copy2(snapshot, context.source_path)
        _append_jsonl(
            context.logs_dir / "actions.jsonl",
            {
                "timestamp": iso_now(),
                "type": "restore_best",
                "iteration": int(best["iteration"]),
                "source_snapshot": str(snapshot),
            },
        )
        print(json.dumps(best, indent=2))
        return 0

    manifest = context.manifest
    restore_script = manifest.get("restore_script")
    if not restore_script:
        print("no restore script configured for this benchmark", file=sys.stderr)
        return 1
    env = os.environ.copy()
    snapshot_env = manifest.get("snapshot_env")
    if snapshot_env:
        env[str(snapshot_env)] = str(context.root_dir / "snapshots")
    cmd = [
        sys.executable,
        str(context.repo_root / restore_script),
        "--script",
        str(context.root_dir / manifest["source_file"]),
        f"--{manifest['benchmark_arg']}",
        str(manifest["benchmark_value"]),
    ]
    proc = subprocess.run(cmd, cwd=context.repo_root, env=env, capture_output=True, text=True, check=False)
    event = {
        "timestamp": iso_now(),
        "type": "restore_best",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    _append_jsonl(context.logs_dir / "actions.jsonl", event)
    if proc.returncode != 0:
        print(proc.stderr.strip() or proc.stdout.strip() or "restore failed", file=sys.stderr)
        return 1
    print(proc.stdout.strip())
    return 0


def _plot_local_run(context: RunContext) -> int:
    import matplotlib.pyplot as plt

    rows = []
    for event in context.evaluations_log.read_text().splitlines() if context.evaluations_log.exists() else []:
        try:
            payload = json.loads(event)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "evaluation":
            continue
        result = payload.get("result") or {}
        if "score" not in result or "iteration" not in payload:
            continue
        rows.append((int(payload["iteration"]), float(result["score"])))
    if not rows:
        raise RuntimeError("no local evaluation log entries to plot")
    rows.sort()
    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    running = []
    best = None
    for value in ys:
        best = value if best is None else min(best, value)
        running.append(best)

    context.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, ys, color="#B6C2CF", linewidth=1.0, alpha=0.8, label="all evaluations")
    ax.plot(xs, running, color="#1f5aa6", linewidth=2.0, label="running best")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title(str(context.manifest.get("display_name", context.root_dir.name)))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    pdf_path = context.figures_dir / "gsopt_score_progress.pdf"
    png_path = context.figures_dir / "gsopt_score_progress.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    payload = {"pdf": str(pdf_path), "png": str(png_path)}
    _append_jsonl(context.logs_dir / "actions.jsonl", {"timestamp": iso_now(), "type": "plot", "result": payload})
    collect_status(context, write=True)
    print(json.dumps(payload, indent=2))
    return 0


def plot_main() -> int:
    context = _context_from_argv0()
    if not context.manifest.get("plot_script") and context.evaluations_log.exists():
        return _plot_local_run(context)

    manifest = context.manifest
    plot_script = manifest.get("plot_script")
    if not plot_script:
        print("no plot script configured for this benchmark", file=sys.stderr)
        return 1
    env = os.environ.copy()
    snapshot_env = manifest.get("snapshot_env")
    fig_dir_env = manifest.get("fig_dir_env")
    run_root_env = manifest.get("run_root_env")
    if snapshot_env:
        env[str(snapshot_env)] = str(context.root_dir / "snapshots")
    if fig_dir_env:
        env[str(fig_dir_env)] = str(context.root_dir / "figs")
    if run_root_env:
        env[str(run_root_env)] = str(context.root_dir / "runs")
    cmd = [sys.executable, str(context.repo_root / plot_script)]
    payload = _subprocess_json(cmd, env=env, cwd=context.repo_root)
    _append_jsonl(
        context.logs_dir / "actions.jsonl",
        {"timestamp": iso_now(), "type": "plot", "command": cmd, "result": payload},
    )
    collect_status(context, write=True)
    print(json.dumps(payload, indent=2))
    return 0


def status_main() -> int:
    context = _context_from_argv0()
    parser = argparse.ArgumentParser(description="Print GSOpt run status.")
    parser.add_argument("--write", action="store_true", help="Rewrite status.json while printing the payload.")
    args = parser.parse_args()
    payload = collect_status(context, write=args.write or context.is_run)
    print(json.dumps(payload, indent=2))
    return 0


def benchmark_optuna_main() -> int:
    context = _context_from_argv0()
    manifest = context.manifest
    optuna_script = manifest.get("optuna_script")
    if not optuna_script:
        raise SystemExit(f"lane {manifest['lane']!r} does not expose an Optuna baseline")

    parser = argparse.ArgumentParser(
        description="Launch the benchmark-local Optuna baseline into a timestamped local archive."
    )
    parser.add_argument(
        "--archive-root",
        help="Optional explicit archive directory. Defaults to <benchmark>/optuna_run_<timestamp>/.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Reuse the newest existing optuna_run_<timestamp> directory instead of creating a fresh one.",
    )
    args, remainder = parser.parse_known_args()

    benchmark_root = (
        Path(context.run_meta["benchmark_root"]).resolve() if context.run_meta is not None else context.root_dir.resolve()
    )
    if args.archive_root:
        archive_root = Path(args.archive_root)
        if not archive_root.is_absolute():
            archive_root = (benchmark_root / archive_root).resolve()
    elif args.resume_latest:
        candidates = sorted(path for path in benchmark_root.glob("optuna_run_*") if path.is_dir())
        if not candidates:
            raise SystemExit(f"no optuna_run_* directories exist under {benchmark_root}")
        archive_root = candidates[-1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = benchmark_root / f"optuna_run_{stamp}"

    archive_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(context.repo_root / optuna_script),
        "--script",
        str(benchmark_root / manifest["source_file"]),
        f"--{manifest['benchmark_arg']}",
        str(manifest["benchmark_value"]),
        "--archive-root",
        str(archive_root),
        *remainder,
    ]
    proc = subprocess.run(cmd, cwd=context.repo_root, check=False)
    return proc.returncode
