from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .campaign_driver import _build_launch_spec, _best_line, _status_line
from .common import iso_now, write_json
from .runtime import RunContext, collect_status, locate_context


def campaign_log_root(context: RunContext, namespace: str = "async") -> Path:
    return context.logs_dir / "campaign" / namespace


def campaign_state_path(context: RunContext, namespace: str = "async") -> Path:
    return campaign_log_root(context, namespace) / "state.json"


def campaign_jsonl_path(context: RunContext, namespace: str = "async") -> Path:
    return campaign_log_root(context, namespace) / "events.jsonl"


def pending_mutation_path(context: RunContext) -> Path:
    return context.logs_dir / "campaign" / "pending_mutation.json"


def append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_state(context: RunContext, payload: dict[str, Any], namespace: str = "async"):
    write_json(campaign_state_path(context, namespace), {"timestamp": iso_now(), **payload})


def _source_hash(context: RunContext) -> str:
    return hashlib.sha256(context.source_path.read_bytes()).hexdigest()


def _eval_command(context: RunContext, description: str) -> list[str]:
    command = [sys.executable, "run_eval.py", "--"]
    if str(context.manifest.get("lane", "")) == "generic":
        command.extend([sys.executable, "evaluate.py", "--description", description])
    else:
        command.extend(["uv", "run", "python", "evaluate.py", "--description", description])
    return command


def _restore_command() -> list[str]:
    return [sys.executable, "restore_best.py"]


def _run(command: list[str], context: RunContext) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=context.root_dir, capture_output=True, text=True, check=False)


def _latest_result(status: dict[str, Any]) -> dict[str, Any]:
    path = status.get("latest_result_path")
    if not path:
        return {}
    return read_json(Path(path))


def _read_pending_description(context: RunContext, fallback: str) -> str:
    payload = read_json(pending_mutation_path(context))
    description = str(payload.get("description") or "").strip()
    if description:
        return description[:240]
    return fallback


def _build_async_prompt(context: RunContext, status: dict[str, Any]) -> str:
    agent_prompt_path = context.root_dir / "agent_prompt.md"
    plan_path = context.root_dir / "plan.md"
    agent_prompt = agent_prompt_path.read_text().strip() if agent_prompt_path.exists() else ""
    plan_text = plan_path.read_text().strip() if plan_path.exists() else ""
    source_file = context.manifest["source_file"]
    run_dir = context.root_dir
    pending_path = pending_mutation_path(context)
    description = context.manifest.get("display_name", context.manifest.get("benchmark_value", run_dir.name))

    return f"""You are waking for one asynchronous GSOpt mutation in `{run_dir}` for `{description}`.

Current campaign state:
- {_status_line(status)}
- {_best_line(status)}
- Live editable file: `{source_file}`

Do exactly this:
- Inspect the latest result and choose the next single mutation.
- Modify `{source_file}` only, unless a small coupled method-support edit is truly required.
- Do not run `evaluate.py`, `run_eval.py`, `restore_best.py`, `campaign.py`, `async_campaign.py`, or `slurm_campaign.py`.
- Write `{pending_path}` as JSON with this exact shape: {{"description": "<short technical mutation summary>"}}.
- Exit after the mutation and summary file are written.

The async driver will score the mutation after you exit, while no agent session is active, then wake a fresh session if more iterations remain.

Key run instructions:
{agent_prompt}

Plan summary:
{plan_text}
"""


def _score_current_source(
    context: RunContext,
    description: str,
    launch_index: int,
    namespace: str,
) -> tuple[bool, dict[str, Any]]:
    write_state(
        context,
        {
            "status": "evaluating",
            "agent_state": "asleep",
            "launch": launch_index,
            "description": description,
        },
        namespace,
    )
    command = _eval_command(context, description)
    started = iso_now()
    append_jsonl(
        campaign_jsonl_path(context, namespace),
        {"timestamp": started, "type": "evaluation_start", "launch": launch_index, "command": command},
    )
    proc = _run(command, context)
    status = collect_status(context, write=True)
    result = _latest_result(status)
    event = {
        "timestamp": iso_now(),
        "type": "evaluation_result",
        "launch": launch_index,
        "returncode": proc.returncode,
        "completed_mutations": status["completed_mutations"],
        "remaining_mutations": status["remaining_mutations"],
        "result_status": result.get("status"),
        "score": result.get("score"),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    append_jsonl(campaign_jsonl_path(context, namespace), event)

    if result.get("status") in {"discard", "crash"} and context.best_path.exists():
        restore = _run(_restore_command(), context)
        append_jsonl(
            campaign_jsonl_path(context, namespace),
            {
                "timestamp": iso_now(),
                "type": "restore_best",
                "returncode": restore.returncode,
                "stdout": restore.stdout.strip(),
                "stderr": restore.stderr.strip(),
            },
        )
    write_state(
        context,
        {
            "status": "waiting" if not status["done"] else "done",
            "agent_state": "asleep",
            "launch": launch_index,
            "completed_mutations": status["completed_mutations"],
            "remaining_mutations": status["remaining_mutations"],
            "best_metric": status.get("best_metric"),
            "last_returncode": proc.returncode,
        },
        namespace,
    )
    return proc.returncode == 0, status


def run_async_step(
    path: Path,
    *,
    agent: str,
    model: str | None,
    search: bool,
    agent_args: list[str],
    launch_index: int,
    namespace: str = "async",
    dry_run: bool = False,
) -> dict[str, Any]:
    context = locate_context(path.resolve())
    if not context.is_run:
        raise SystemExit("async campaign mode requires a gsopt run directory; scaffold a run first")

    status_before = collect_status(context, write=True)
    if status_before["done"]:
        write_state(context, {"status": "done", "agent": agent, "launch": launch_index}, namespace)
        return {"done": True, "progressed": False, "status": status_before}

    if status_before.get("latest_iteration") is None:
        if dry_run:
            return {
                "done": False,
                "progressed": False,
                "dry_run": True,
                "would": "score_baseline",
                "command": _eval_command(context, "archive untouched baseline"),
            }
        ok, status_after = _score_current_source(context, "archive untouched baseline", launch_index, namespace)
        return {"done": status_after["done"], "progressed": ok, "status": status_after}

    pending = pending_mutation_path(context)
    pending.parent.mkdir(parents=True, exist_ok=True)
    if pending.exists():
        pending.unlink()
    before_hash = _source_hash(context)
    prompt = _build_async_prompt(context, status_before)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = campaign_log_root(context, namespace)
    log_root.mkdir(parents=True, exist_ok=True)
    log_prefix = log_root / f"launch_{launch_index:04d}_{stamp}"
    log_prefix.with_suffix(".prompt.txt").write_text(prompt)
    spec = _build_launch_spec(context, agent, prompt, model, search, agent_args, log_prefix)
    append_jsonl(
        campaign_jsonl_path(context, namespace),
        {
            "timestamp": iso_now(),
            "type": "agent_launch",
            "launch": launch_index,
            "agent": agent,
            "model": model,
            "command": spec.command,
            "completed_mutations": status_before["completed_mutations"],
        },
    )
    write_state(
        context,
        {
            "status": "agent_running",
            "agent_state": "awake",
            "agent": agent,
            "model": model,
            "launch": launch_index,
            "completed_mutations": status_before["completed_mutations"],
            "remaining_mutations": status_before["remaining_mutations"],
        },
        namespace,
    )

    if dry_run:
        return {"done": False, "progressed": False, "dry_run": True, "command": spec.command, "prompt": prompt}

    proc = _run(spec.command, context)
    log_prefix.with_suffix(".stdout.txt").write_text(proc.stdout)
    log_prefix.with_suffix(".stderr.txt").write_text(proc.stderr)
    after_hash = _source_hash(context)
    changed = after_hash != before_hash
    description = _read_pending_description(context, f"async mutation launch {launch_index}")
    append_jsonl(
        campaign_jsonl_path(context, namespace),
        {
            "timestamp": iso_now(),
            "type": "agent_result",
            "launch": launch_index,
            "returncode": proc.returncode,
            "source_changed": changed,
            "description": description,
            "stdout_path": str(log_prefix.with_suffix(".stdout.txt")),
            "stderr_path": str(log_prefix.with_suffix(".stderr.txt")),
        },
    )
    if not changed:
        status = collect_status(context, write=True)
        write_state(
            context,
            {
                "status": "stalled",
                "reason": "agent_made_no_source_change",
                "agent_state": "asleep",
                "launch": launch_index,
                "completed_mutations": status["completed_mutations"],
                "remaining_mutations": status["remaining_mutations"],
            },
            namespace,
        )
        return {"done": False, "progressed": False, "status": status, "reason": "no_source_change"}

    ok, status_after = _score_current_source(context, description, launch_index, namespace)
    return {"done": status_after["done"], "progressed": ok, "status": status_after}


def run_async_campaign(
    path: Path,
    agent: str,
    model: str | None,
    max_launches: int,
    sleep_seconds: float,
    stall_launches: int,
    search: bool,
    agent_args: list[str],
    dry_run: bool,
) -> int:
    context = locate_context(path.resolve())
    no_progress = 0
    for launch in range(1, max_launches + 1):
        result = run_async_step(
            context.root_dir,
            agent=agent,
            model=model,
            search=search,
            agent_args=agent_args,
            launch_index=launch,
            namespace="async",
            dry_run=dry_run,
        )
        print(json.dumps(result, indent=2), flush=True)
        if dry_run or result.get("done"):
            return 0
        if result.get("progressed"):
            no_progress = 0
        else:
            no_progress += 1
            if no_progress >= stall_launches:
                write_state(
                    context,
                    {
                        "status": "stalled",
                        "reason": result.get("reason", "no_progress"),
                        "agent": agent,
                        "launches": launch,
                    },
                    "async",
                )
                return 1
        time.sleep(sleep_seconds)

    status = collect_status(context, write=True)
    write_state(
        context,
        {
            "status": "done" if status["done"] else "incomplete",
            "agent": agent,
            "launches": max_launches,
            "completed_mutations": status["completed_mutations"],
            "remaining_mutations": status["remaining_mutations"],
        },
        "async",
    )
    return 0 if status["done"] else 1
