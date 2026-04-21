from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _SCRIPTS_ROOT = Path(__file__).resolve().parent
    if str(_SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_ROOT))
    from gsopt_runtime.campaign_driver import run_campaign
    from gsopt_runtime.campaign_watchdog import run_watchdog
    from gsopt_runtime.common import find_skill_root
    from gsopt_runtime.runtime import collect_status, locate_context
    from gsopt_runtime.scaffold import init_run, sync_benchmark_entrypoints
    from gsopt_runtime.slurm_campaign import run_slurm_step, submit_slurm_campaign
    from gsopt_runtime.tui import run_tui
else:
    from .gsopt_runtime.campaign_driver import run_campaign
    from .gsopt_runtime.campaign_watchdog import run_watchdog
    from .gsopt_runtime.common import find_skill_root
    from .gsopt_runtime.runtime import collect_status, locate_context
    from .gsopt_runtime.scaffold import init_run, sync_benchmark_entrypoints
    from .gsopt_runtime.slurm_campaign import run_slurm_step, submit_slurm_campaign
    from .gsopt_runtime.tui import run_tui


def _run_eval_wrapper(command: list[str], run_dir: Path | None):
    if not command:
        raise SystemExit("run-eval requires a command after `--`")
    skill_root = find_skill_root()
    script = skill_root / "scripts" / "run_gsopt.sh"
    env = os.environ.copy()
    env["GSOPT_RUNTIME_ROOT"] = str(skill_root)
    if run_dir is not None:
        env["GSOPT_RUN_DIR"] = str(run_dir)
    proc = subprocess.run(["bash", str(script), *command], cwd=run_dir or Path.cwd(), env=env, check=False)
    raise SystemExit(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold and monitor reproducible energy-optimization runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-run", help="Create a timestamped run directory from a canonical benchmark.")
    init_parser.add_argument("iterations", type=int)
    init_parser.add_argument("target", nargs="?", help="Benchmark dir, benchmark file, live source file, or current dir.")
    init_parser.add_argument(
        "--source",
        default=None,
        help="Editable source file relative to the benchmark dir when it cannot be inferred automatically.",
    )
    init_parser.add_argument(
        "--evaluator",
        default=None,
        help="Evaluator file relative to the benchmark dir when it is not named evaluate.py/evaluator.py/eval.py.",
    )
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
        help="Refresh benchmark-local evaluate.py wrappers from existing manifests.",
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

    tui_parser = subparsers.add_parser("tui", help="Open a live terminal monitor for a GSOpt run.")
    tui_parser.add_argument("path", nargs="?", default=".")
    tui_parser.add_argument("--refresh-seconds", type=float, default=2.0)
    tui_parser.add_argument("--once", action="store_true", help="Print one dashboard frame and exit.")

    campaign_parser = subparsers.add_parser(
        "campaign",
        help="Repeatedly relaunch Codex or Claude until the run reaches its target iteration count.",
    )
    campaign_parser.add_argument("path", nargs="?", default=".")
    campaign_parser.add_argument("--agent", required=True, choices=("codex", "claude"))
    campaign_parser.add_argument("--model", default=None)
    campaign_parser.add_argument("--max-launches", type=int, default=50)
    campaign_parser.add_argument("--sleep-seconds", type=float, default=5.0)
    campaign_parser.add_argument("--stall-launches", type=int, default=3)
    campaign_parser.add_argument("--search", action="store_true", help="Enable web search for Codex launches.")
    campaign_parser.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="Extra raw argument passed through to the underlying agent CLI. Repeatable.",
    )
    campaign_parser.add_argument("--dry-run", action="store_true")

    slurm_parser = subparsers.add_parser(
        "slurm-campaign",
        help="Submit a self-resubmitting Slurm campaign that relaunches Codex or Claude one job at a time.",
    )
    slurm_parser.add_argument("path", nargs="?", default=".")
    slurm_parser.add_argument("--agent", required=True, choices=("codex", "claude"))
    slurm_parser.add_argument("--model", default=None)
    slurm_parser.add_argument("--max-launches", type=int, default=50)
    slurm_parser.add_argument("--sleep-seconds", type=float, default=5.0)
    slurm_parser.add_argument("--stall-launches", type=int, default=3)
    slurm_parser.add_argument("--search", action="store_true", help="Enable web search for agent launches when supported.")
    slurm_parser.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="Extra raw argument passed through to the underlying agent CLI. Repeatable.",
    )
    slurm_parser.add_argument("--partition", default=None)
    slurm_parser.add_argument("--account", default=None)
    slurm_parser.add_argument("--qos", default=None)
    slurm_parser.add_argument("--time", dest="time_limit", default="04:00:00")
    slurm_parser.add_argument("--cpus-per-task", type=int, default=None)
    slurm_parser.add_argument("--mem", default=None)
    slurm_parser.add_argument("--gres", default=None)
    slurm_parser.add_argument("--constraint", default=None)
    slurm_parser.add_argument("--job-name", default=None)
    slurm_parser.add_argument(
        "--setup-command",
        action="append",
        default=[],
        help="Shell line inserted before the agent launch in each Slurm job. Repeatable.",
    )
    slurm_parser.add_argument(
        "--sbatch-directive",
        action="append",
        default=[],
        help="Additional raw #SBATCH directive, for example '--mail-type=FAIL'. Repeatable.",
    )
    slurm_parser.add_argument("--dry-run", action="store_true")
    slurm_parser.add_argument("--force", action="store_true", help="Submit even if state says a Slurm campaign is active.")

    slurm_step = subparsers.add_parser("slurm-step", help="Internal worker invoked by a GSOpt Slurm campaign job.")
    slurm_step.add_argument("path", nargs="?", default=".")
    slurm_step.add_argument("--agent", required=True, choices=("codex", "claude"))
    slurm_step.add_argument("--model", default=None)
    slurm_step.add_argument("--max-launches", type=int, default=50)
    slurm_step.add_argument("--sleep-seconds", type=float, default=5.0)
    slurm_step.add_argument("--stall-launches", type=int, default=3)
    slurm_step.add_argument("--search", action="store_true")
    slurm_step.add_argument("--agent-arg", action="append", default=[])

    run_eval = subparsers.add_parser("run-eval", help="Wrap a command with heartbeat and worker watchdog logs.")
    run_eval.add_argument("--run-dir", default=".")
    run_eval.add_argument("cmd", nargs=argparse.REMAINDER)

    return parser


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0].isdigit():
        argv = ["init-run", *argv]
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    if extras:
        if args.command == "init-run" and all(not token.startswith("-") for token in extras):
            args.instructions.extend(extras)
        else:
            parser.error(f"unrecognized arguments: {' '.join(extras)}")

    if args.command == "init-run":
        max_parallel = args.max_parallel
        if args.evaluation_mode == "serialized":
            max_parallel = 1
        try:
            payload = init_run(
                args.iterations,
                args.target,
                " ".join(args.instructions).strip(),
                args.evaluation_mode,
                max_parallel,
                source_hint=args.source,
                evaluator_hint=args.evaluator,
            )
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
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

    if args.command == "tui":
        return run_tui(Path(args.path).resolve(), args.refresh_seconds, args.once)

    if args.command == "campaign":
        return run_campaign(
            Path(args.path).resolve(),
            args.agent,
            args.model,
            args.max_launches,
            args.sleep_seconds,
            args.stall_launches,
            args.search,
            list(args.agent_arg),
            args.dry_run,
        )

    if args.command == "slurm-campaign":
        return submit_slurm_campaign(
            Path(args.path).resolve(),
            agent=args.agent,
            model=args.model,
            max_launches=args.max_launches,
            sleep_seconds=args.sleep_seconds,
            stall_launches=args.stall_launches,
            search=args.search,
            agent_args=list(args.agent_arg),
            partition=args.partition,
            account=args.account,
            qos=args.qos,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            gres=args.gres,
            constraint=args.constraint,
            job_name=args.job_name,
            setup_commands=list(args.setup_command),
            sbatch_directives=list(args.sbatch_directive),
            dry_run=args.dry_run,
            force=args.force,
        )

    if args.command == "slurm-step":
        return run_slurm_step(
            Path(args.path).resolve(),
            agent=args.agent,
            model=args.model,
            max_launches=args.max_launches,
            sleep_seconds=args.sleep_seconds,
            stall_launches=args.stall_launches,
            search=args.search,
            agent_args=list(args.agent_arg),
        )

    if args.command == "run-eval":
        command = list(args.cmd)
        if command and command[0] == "--":
            command = command[1:]
        _run_eval_wrapper(command, Path(args.run_dir).resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
