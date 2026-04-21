from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .campaign_driver import _build_launch_spec, _build_prompt
from .common import find_skill_root, iso_now, write_json
from .runtime import RunContext, collect_status, locate_context


def _slurm_root(context: RunContext) -> Path:
    return context.logs_dir / "campaign" / "slurm"


def _state_path(context: RunContext) -> Path:
    return _slurm_root(context) / "slurm_state.json"


def _config_path(context: RunContext) -> Path:
    return _slurm_root(context) / "slurm_config.json"


def _jsonl_path(context: RunContext) -> Path:
    return _slurm_root(context) / "slurm_campaign.jsonl"


def _script_path(context: RunContext) -> Path:
    return _slurm_root(context) / "gsopt_slurm_step.sbatch"


def _append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _option(args: list[str], name: str, value: str | int | float | None):
    if value is None:
        return
    args.extend([name, str(value)])


def _flag(args: list[str], name: str, enabled: bool):
    if enabled:
        args.append(name)


def _build_step_args(context: RunContext, config: dict[str, Any]) -> list[str]:
    skill_root = find_skill_root()
    script = skill_root / "scripts" / "gsopt_cli.py"
    args = [
        "python3",
        str(script),
        "slurm-step",
        str(context.root_dir),
        "--agent",
        str(config["agent"]),
        "--max-launches",
        str(config["max_launches"]),
        "--stall-launches",
        str(config["stall_launches"]),
        "--sleep-seconds",
        str(config["sleep_seconds"]),
    ]
    _option(args, "--model", config.get("model"))
    _flag(args, "--search", bool(config.get("search")))
    for agent_arg in config.get("agent_args", []):
        args.extend(["--agent-arg", str(agent_arg)])
    return args


def _sbatch_line(name: str, value: str | int | None) -> str | None:
    if value is None or value == "":
        return None
    return f"#SBATCH --{name}={value}"


def _write_batch_script(context: RunContext, config: dict[str, Any]) -> Path:
    root = _slurm_root(context)
    root.mkdir(parents=True, exist_ok=True)
    setup_lines = [str(line) for line in config.get("setup_commands", []) if str(line).strip()]
    step_args = " ".join(shlex.quote(part) for part in _build_step_args(context, config))
    directives = [
        "#!/usr/bin/env bash",
        _sbatch_line("job-name", config.get("job_name")),
        _sbatch_line("output", str(root / "slurm-%j.out")),
        _sbatch_line("error", str(root / "slurm-%j.err")),
        _sbatch_line("time", config.get("time")),
        _sbatch_line("partition", config.get("partition")),
        _sbatch_line("account", config.get("account")),
        _sbatch_line("qos", config.get("qos")),
        _sbatch_line("cpus-per-task", config.get("cpus_per_task")),
        _sbatch_line("mem", config.get("mem")),
        _sbatch_line("gres", config.get("gres")),
        _sbatch_line("constraint", config.get("constraint")),
    ]
    for raw in config.get("sbatch_directives", []):
        raw_text = str(raw).strip()
        if raw_text:
            directives.append(raw_text if raw_text.startswith("#SBATCH") else f"#SBATCH {raw_text}")
    body = [line for line in directives if line]
    body.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(context.root_dir))}",
            *setup_lines,
            f"export GSOPT_SLURM_RUN_DIR={shlex.quote(str(context.root_dir))}",
            f"export GSOPT_SLURM_AGENT={shlex.quote(str(config['agent']))}",
            step_args,
            "",
        ]
    )
    script_path = _script_path(context)
    script_path.write_text("\n".join(body))
    script_path.chmod(0o755)
    return script_path


def _submit_script(context: RunContext, config: dict[str, Any], reason: str, dry_run: bool) -> dict[str, Any]:
    script_path = _write_batch_script(context, config)
    event = {
        "timestamp": iso_now(),
        "type": "submit",
        "reason": reason,
        "script": str(script_path),
        "agent": config["agent"],
    }
    if dry_run:
        event["dry_run"] = True
        event["script_text"] = script_path.read_text()
        _append_jsonl(_jsonl_path(context), event)
        return event

    proc = subprocess.run(["sbatch", "--parsable", str(script_path)], cwd=context.root_dir, capture_output=True, text=True, check=False)
    event["returncode"] = proc.returncode
    event["stdout"] = proc.stdout.strip()
    event["stderr"] = proc.stderr.strip()
    if proc.returncode != 0:
        error = proc.stderr.strip() or proc.stdout.strip() or "sbatch failed"
        event["error"] = error
        _append_jsonl(_jsonl_path(context), event)
        write_json(
            _state_path(context),
            {
                "timestamp": iso_now(),
                "status": "submit_failed",
                "agent": config["agent"],
                "error": error,
            },
        )
        raise RuntimeError(error)

    job_id = proc.stdout.strip().split(";", 1)[0].strip()
    event["job_id"] = job_id
    _append_jsonl(_jsonl_path(context), event)
    state = _read_json(_state_path(context))
    write_json(
        _state_path(context),
        {
            **state,
            "timestamp": iso_now(),
            "status": "submitted",
            "agent": config["agent"],
            "model": config.get("model"),
            "job_id": job_id,
            "run_dir": str(context.root_dir),
            "max_launches": config["max_launches"],
        },
    )
    return event


def _config_payload(
    *,
    agent: str,
    model: str | None,
    max_launches: int,
    sleep_seconds: float,
    stall_launches: int,
    search: bool,
    agent_args: list[str],
    partition: str | None,
    account: str | None,
    qos: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    mem: str | None,
    gres: str | None,
    constraint: str | None,
    job_name: str | None,
    setup_commands: list[str],
    sbatch_directives: list[str],
) -> dict[str, Any]:
    return {
        "agent": agent,
        "model": model,
        "max_launches": max_launches,
        "sleep_seconds": sleep_seconds,
        "stall_launches": stall_launches,
        "search": search,
        "agent_args": list(agent_args),
        "partition": partition,
        "account": account,
        "qos": qos,
        "time": time_limit,
        "cpus_per_task": cpus_per_task,
        "mem": mem,
        "gres": gres,
        "constraint": constraint,
        "job_name": job_name,
        "setup_commands": list(setup_commands),
        "sbatch_directives": list(sbatch_directives),
    }


def submit_slurm_campaign(
    path: Path,
    *,
    agent: str,
    model: str | None,
    max_launches: int,
    sleep_seconds: float,
    stall_launches: int,
    search: bool,
    agent_args: list[str],
    partition: str | None,
    account: str | None,
    qos: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    mem: str | None,
    gres: str | None,
    constraint: str | None,
    job_name: str | None,
    setup_commands: list[str],
    sbatch_directives: list[str],
    dry_run: bool,
    force: bool,
) -> int:
    context = locate_context(path.resolve())
    if not context.is_run:
        raise SystemExit("slurm campaign mode requires a gsopt run directory; scaffold a run first")
    status = collect_status(context, write=True)
    if status["done"]:
        print(json.dumps({"done": True, "run_dir": str(context.root_dir), "status": status}, indent=2))
        return 0

    state = _read_json(_state_path(context))
    if state.get("status") in {"submitted", "running"} and not force and not dry_run:
        raise SystemExit(
            f"slurm campaign already {state['status']} with job {state.get('job_id', '-')}; "
            "rerun with --force to submit another job"
        )

    config = _config_payload(
        agent=agent,
        model=model,
        max_launches=max_launches,
        sleep_seconds=sleep_seconds,
        stall_launches=stall_launches,
        search=search,
        agent_args=agent_args,
        partition=partition,
        account=account,
        qos=qos,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        mem=mem,
        gres=gres,
        constraint=constraint,
        job_name=job_name or f"gsopt-{context.manifest.get('benchmark_value', context.root_dir.name)}",
        setup_commands=setup_commands,
        sbatch_directives=sbatch_directives,
    )
    _slurm_root(context).mkdir(parents=True, exist_ok=True)
    write_json(_config_path(context), config)
    event = _submit_script(context, config, "start", dry_run=dry_run)
    if dry_run:
        print(json.dumps(event, indent=2))
        return 0
    print(json.dumps({"submitted": True, "job_id": event.get("job_id"), "run_dir": str(context.root_dir)}, indent=2))
    return 0


def run_slurm_step(
    path: Path,
    *,
    agent: str,
    model: str | None,
    max_launches: int,
    sleep_seconds: float,
    stall_launches: int,
    search: bool,
    agent_args: list[str],
) -> int:
    context = locate_context(path.resolve())
    if not context.is_run:
        raise SystemExit("slurm-step requires a gsopt run directory")

    persisted_config = _read_json(_config_path(context))
    config = {
        **persisted_config,
        "agent": agent,
        "model": model,
        "max_launches": max_launches,
        "sleep_seconds": sleep_seconds,
        "stall_launches": stall_launches,
        "search": search,
        "agent_args": agent_args,
    }
    state_before = _read_json(_state_path(context))
    launch_count = int(state_before.get("launch_count", 0)) + 1
    no_progress_count = int(state_before.get("no_progress_count", 0))
    job_id = os.environ.get("SLURM_JOB_ID", "")
    status_before = collect_status(context, write=True)
    if status_before["done"]:
        write_json(_state_path(context), {**state_before, "timestamp": iso_now(), "status": "done"})
        print(json.dumps({"done": True, "run_dir": str(context.root_dir), "status": status_before}, indent=2))
        return 0

    prompt = _build_prompt(context, status_before)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = _slurm_root(context)
    log_root.mkdir(parents=True, exist_ok=True)
    log_prefix = log_root / f"launch_{launch_count:04d}_{stamp}"
    log_prefix.with_suffix(".prompt.txt").write_text(prompt)

    write_json(
        _state_path(context),
        {
            **state_before,
            "timestamp": iso_now(),
            "status": "running",
            "agent": agent,
            "model": model,
            "job_id": job_id,
            "launch_count": launch_count,
            "no_progress_count": no_progress_count,
            "completed_mutations": status_before["completed_mutations"],
            "remaining_mutations": status_before["remaining_mutations"],
        },
    )
    _append_jsonl(
        _jsonl_path(context),
        {
            "timestamp": iso_now(),
            "type": "launch",
            "launch": launch_count,
            "job_id": job_id,
            "agent": agent,
            "model": model,
            "status_before": {
                "completed_mutations": status_before["completed_mutations"],
                "latest_iteration": status_before.get("latest_iteration"),
                "remaining_mutations": status_before["remaining_mutations"],
            },
        },
    )

    spec = _build_launch_spec(context, agent, prompt, model, search, agent_args, log_prefix)
    proc = subprocess.run(spec.command, cwd=context.root_dir, capture_output=True, text=True, check=False)
    log_prefix.with_suffix(".stdout.txt").write_text(proc.stdout)
    log_prefix.with_suffix(".stderr.txt").write_text(proc.stderr)

    status_after = collect_status(context, write=True)
    completed_before = int(status_before["completed_mutations"])
    completed_after = int(status_after["completed_mutations"])
    progressed = completed_after > completed_before
    no_progress_count = 0 if progressed else no_progress_count + 1
    summary = {
        "timestamp": iso_now(),
        "type": "launch_result",
        "launch": launch_count,
        "job_id": job_id,
        "agent": agent,
        "returncode": proc.returncode,
        "completed_mutations_before": completed_before,
        "completed_mutations_after": completed_after,
        "latest_iteration_before": status_before.get("latest_iteration"),
        "latest_iteration_after": status_after.get("latest_iteration"),
        "done": status_after["done"],
        "progressed": progressed,
        "stdout_path": str(log_prefix.with_suffix(".stdout.txt")),
        "stderr_path": str(log_prefix.with_suffix(".stderr.txt")),
    }
    _append_jsonl(_jsonl_path(context), summary)

    if status_after["done"]:
        write_json(
            _state_path(context),
            {
                "timestamp": iso_now(),
                "status": "done",
                "agent": agent,
                "model": model,
                "job_id": job_id,
                "launch_count": launch_count,
                "no_progress_count": no_progress_count,
                "completed_mutations": completed_after,
                "remaining_mutations": status_after["remaining_mutations"],
                "last_returncode": proc.returncode,
            },
        )
        print(json.dumps(summary, indent=2))
        return 0

    if launch_count >= max_launches:
        failure = {
            "timestamp": iso_now(),
            "status": "incomplete",
            "reason": "max_launches",
            "agent": agent,
            "launch_count": launch_count,
            "completed_mutations": completed_after,
            "remaining_mutations": status_after["remaining_mutations"],
        }
        _append_jsonl(_jsonl_path(context), failure)
        write_json(_state_path(context), failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1

    if no_progress_count >= stall_launches:
        failure = {
            "timestamp": iso_now(),
            "status": "stalled",
            "reason": "no_progress",
            "agent": agent,
            "launch_count": launch_count,
            "completed_mutations": completed_after,
            "remaining_mutations": status_after["remaining_mutations"],
        }
        _append_jsonl(_jsonl_path(context), failure)
        write_json(_state_path(context), failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1

    write_json(
        _state_path(context),
        {
            "timestamp": iso_now(),
            "status": "resubmitting",
            "agent": agent,
            "model": model,
            "job_id": job_id,
            "launch_count": launch_count,
            "no_progress_count": no_progress_count,
            "completed_mutations": completed_after,
            "remaining_mutations": status_after["remaining_mutations"],
            "last_returncode": proc.returncode,
        },
    )
    if sleep_seconds > 0:
        import time

        time.sleep(sleep_seconds)
    event = _submit_script(context, config, "step_complete", dry_run=False)
    print(json.dumps({"resubmitted": True, "next_job_id": event.get("job_id"), **summary}, indent=2))
    return 0
