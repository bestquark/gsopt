from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import find_repo_root
from .runtime import collect_status, locate_context
from .scaffold import init_run, sync_benchmark_entrypoints
from .watchdog import run_watchdog


def _run_eval_wrapper(command: list[str], run_dir: Path | None):
    if not command:
        raise SystemExit("run-eval requires a command after `--`")
    repo_root = find_repo_root()
    script = repo_root / "skills" / "gsopt" / "scripts" / "run_gsopt.sh"
    env = os.environ.copy()
    if run_dir is not None:
        env["GSOPT_RUN_DIR"] = str(run_dir)
    proc = subprocess.run(["bash", str(script), *command], cwd=run_dir or repo_root, env=env, check=False)
    raise SystemExit(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold and monitor reproducible energy-optimization runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-run", help="Create a timestamped run directory from a canonical benchmark.")
    init_parser.add_argument("iterations", type=int)
    init_parser.add_argument("target", nargs="?", help="Template dir, template file, live source file, or current dir.")
    init_parser.add_argument(
        "--evaluation-mode",
        choices=("serialized", "parallel"),
        default="serialized",
        help="Whether scored evaluations should be serialized or allow limited queue parallelism.",
    )
    init_parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent queued evaluations allowed for this run.",
    )
    init_parser.add_argument("instructions", nargs="*", help="Additional instructions appended to the run prompt.")

    subparsers.add_parser(
        "sync-benchmarks",
        help="Refresh benchmark-local .gsopt.json, evaluate.py, and optuna_baseline.py wrappers.",
    )

    status_parser = subparsers.add_parser("status", help="Print run status for a run dir or current context.")
    status_parser.add_argument("path", nargs="?", default=".")
    status_parser.add_argument("--write", action="store_true")

    render_parser = subparsers.add_parser("render", help="Render figures for a run dir or current context.")
    render_parser.add_argument("path", nargs="?", default=".")

    watchdog_parser = subparsers.add_parser("watchdog", help="Monitor a run until the target iteration count is reached.")
    watchdog_parser.add_argument("path", nargs="?", default=".")
    watchdog_parser.add_argument("--poll-seconds", type=float, default=20.0)
    watchdog_parser.add_argument("--stall-seconds", type=float, default=900.0)

    run_eval = subparsers.add_parser("run-eval", help="Wrap a command with heartbeat and worker watchdog logs.")
    run_eval.add_argument("--run-dir", default=".")
    run_eval.add_argument("cmd", nargs=argparse.REMAINDER)

    return parser


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0].isdigit():
        argv = ["init-run", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-run":
        max_parallel = args.max_parallel
        if args.evaluation_mode == "serialized":
            max_parallel = 1
        payload = init_run(
            args.iterations,
            args.target,
            " ".join(args.instructions).strip(),
            args.evaluation_mode,
            max_parallel,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "sync-benchmarks":
        print(json.dumps(sync_benchmark_entrypoints(), indent=2))
        return 0

    if args.command == "status":
        context = locate_context(Path(args.path).resolve())
        print(json.dumps(collect_status(context, write=args.write or context.is_run), indent=2))
        return 0

    if args.command == "render":
        context = locate_context(Path(args.path).resolve())
        plot_script = context.root_dir / "plot.py"
        proc = subprocess.run([sys.executable, str(plot_script)], cwd=context.root_dir, check=False)
        return proc.returncode

    if args.command == "watchdog":
        return run_watchdog(Path(args.path).resolve(), args.poll_seconds, args.stall_seconds)

    if args.command == "run-eval":
        command = list(args.cmd)
        if command and command[0] == "--":
            command = command[1:]
        _run_eval_wrapper(command, Path(args.run_dir).resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
