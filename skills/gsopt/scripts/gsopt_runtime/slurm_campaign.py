from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .async_campaign import read_json, run_async_step, write_state
from .common import find_skill_root, iso_now, write_json
from .runtime import collect_status, locate_context


def _root(run_dir: Path) -> Path:
    return run_dir / "logs" / "campaign" / "slurm"


def _config_path(run_dir: Path) -> Path:
    return _root(run_dir) / "config.json"


def _script_path(run_dir: Path) -> Path:
    return _root(run_dir) / "step.sbatch"


def _jsonl_path(run_dir: Path) -> Path:
    return _root(run_dir) / "events.jsonl"


def _append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _add_option(args: list[str], flag: str, value: Any):
    if value is not None:
        args.extend([flag, str(value)])


def _step_command(run_dir: Path, config: dict[str, Any]) -> list[str]:
    cli = find_skill_root() / "scripts" / "gsopt_cli.py"
    args = [
        sys.executable,
        str(cli),
        "slurm-step",
        str(run_dir),
        "--agent",
        str(config["agent"]),
        "--max-launches",
        str(config["max_launches"]),
        "--stall-launches",
        str(config["stall_launches"]),
    ]
    _add_option(args, "--model", config.get("model"))
    if config.get("search"):
        args.append("--search")
    for agent_arg in config.get("agent_args", []):
        args.extend(["--agent-arg", str(agent_arg)])
    return args


def _directive(name: str, value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return f"#SBATCH --{name}={value}"


def _write_script(run_dir: Path, config: dict[str, Any]) -> Path:
    root = _root(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    directives = [
        "#!/usr/bin/env bash",
        _directive("job-name", config.get("job_name")),
        _directive("output", str(root / "slurm-%j.out")),
        _directive("error", str(root / "slurm-%j.err")),
        _directive("time", config.get("time")),
        _directive("partition", config.get("partition")),
        _directive("account", config.get("account")),
        _directive("qos", config.get("qos")),
        _directive("cpus-per-task", config.get("cpus_per_task")),
        _directive("mem", config.get("mem")),
        _directive("gres", config.get("gres")),
        _directive("constraint", config.get("constraint")),
    ]
    for raw in config.get("sbatch_directives", []):
        text = str(raw).strip()
        if text:
            directives.append(text if text.startswith("#SBATCH") else f"#SBATCH {text}")

    setup = [str(line) for line in config.get("setup_commands", []) if str(line).strip()]
    body = [line for line in directives if line]
    body.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(run_dir))}",
            *setup,
            " ".join(shlex.quote(part) for part in _step_command(run_dir, config)),
            "",
        ]
    )
    script = _script_path(run_dir)
    script.write_text("\n".join(body))
    script.chmod(0o755)
    return script


def _submit(run_dir: Path, config: dict[str, Any], reason: str, dry_run: bool) -> dict[str, Any]:
    script = _write_script(run_dir, config)
    event: dict[str, Any] = {
        "timestamp": iso_now(),
        "type": "submit",
        "reason": reason,
        "script": str(script),
        "agent": config["agent"],
    }
    if dry_run:
        event["dry_run"] = True
        event["script_text"] = script.read_text()
        _append_jsonl(_jsonl_path(run_dir), event)
        return event

    proc = subprocess.run(["sbatch", "--parsable", str(script)], cwd=run_dir, capture_output=True, text=True, check=False)
    event.update({"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
    if proc.returncode != 0:
        event["error"] = proc.stderr.strip() or proc.stdout.strip() or "sbatch failed"
        _append_jsonl(_jsonl_path(run_dir), event)
        write_state(locate_context(run_dir), {"status": "submit_failed", "error": event["error"]}, "slurm")
        raise RuntimeError(str(event["error"]))

    event["job_id"] = proc.stdout.strip().split(";", 1)[0].strip()
    _append_jsonl(_jsonl_path(run_dir), event)
    previous = read_json(_root(run_dir) / "state.json")
    write_state(
        locate_context(run_dir),
        {
            "status": "submitted",
            "agent_state": "asleep",
            "agent": config["agent"],
            "model": config.get("model"),
            "job_id": event["job_id"],
            "launch": previous.get("launch", 0),
            "no_progress_count": previous.get("no_progress_count", 0),
        },
        "slurm",
    )
    return event


def _config(args: dict[str, Any], context_run_dir: Path) -> dict[str, Any]:
    return {
        "agent": args["agent"],
        "model": args.get("model"),
        "max_launches": args["max_launches"],
        "stall_launches": args["stall_launches"],
        "search": args["search"],
        "agent_args": list(args.get("agent_args", [])),
        "partition": args.get("partition"),
        "account": args.get("account"),
        "qos": args.get("qos"),
        "time": args.get("time_limit"),
        "cpus_per_task": args.get("cpus_per_task"),
        "mem": args.get("mem"),
        "gres": args.get("gres"),
        "constraint": args.get("constraint"),
        "job_name": args.get("job_name") or f"gsopt-{context_run_dir.name}",
        "setup_commands": list(args.get("setup_commands", [])),
        "sbatch_directives": list(args.get("sbatch_directives", [])),
    }


def submit_slurm_campaign(path: Path, **kwargs: Any) -> int:
    context = locate_context(path.resolve())
    if not context.is_run:
        raise SystemExit("slurm campaign mode requires a gsopt run directory; scaffold a run first")
    status = collect_status(context, write=True)
    if status["done"]:
        print(json.dumps({"done": True, "run_dir": str(context.root_dir), "status": status}, indent=2))
        return 0

    config = _config(kwargs, context.root_dir)
    state = read_json(_root(context.root_dir) / "state.json")
    if state.get("status") in {"submitted", "agent_running", "evaluating"} and not kwargs.get("force"):
        raise SystemExit("a Slurm GSOpt campaign already appears active; use --force to submit anyway")

    _root(context.root_dir).mkdir(parents=True, exist_ok=True)
    write_json(_config_path(context.root_dir), config)
    event = _submit(context.root_dir, config, "start", bool(kwargs.get("dry_run")))
    print(json.dumps(event, indent=2))
    return 0


def run_slurm_step(
    path: Path,
    *,
    agent: str,
    model: str | None,
    max_launches: int,
    stall_launches: int,
    search: bool,
    agent_args: list[str],
) -> int:
    context = locate_context(path.resolve())
    config = {
        **read_json(_config_path(context.root_dir)),
        "agent": agent,
        "model": model,
        "max_launches": max_launches,
        "stall_launches": stall_launches,
        "search": search,
        "agent_args": agent_args,
    }
    state = read_json(_root(context.root_dir) / "state.json")
    launch = int(state.get("launch", 0)) + 1
    if launch > max_launches:
        write_state(context, {"status": "incomplete", "reason": "max_launches", "launch": launch - 1}, "slurm")
        return 1

    result = run_async_step(
        context.root_dir,
        agent=agent,
        model=model,
        search=search,
        agent_args=agent_args,
        launch_index=launch,
        namespace="slurm",
    )
    status = collect_status(context, write=True)
    if result.get("done") or status["done"]:
        write_state(context, {"status": "done", "agent": agent, "launch": launch}, "slurm")
        print(json.dumps(result, indent=2))
        return 0
    if not result.get("progressed"):
        no_progress = int(state.get("no_progress_count", 0)) + 1
        if no_progress >= stall_launches:
            write_state(context, {"status": "stalled", "reason": result.get("reason", "no_progress"), "launch": launch}, "slurm")
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 1
        write_state(context, {"status": "waiting", "no_progress_count": no_progress, "launch": launch}, "slurm")

    event = _submit(context.root_dir, config, "step_complete", dry_run=False)
    print(json.dumps({"resubmitted": event.get("job_id"), "step": result}, indent=2))
    return 0
