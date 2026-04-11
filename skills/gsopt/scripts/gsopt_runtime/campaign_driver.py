from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .common import iso_now, write_json
from .runtime import RunContext, collect_status, locate_context


@dataclass(frozen=True)
class LaunchSpec:
    name: str
    command: list[str]


def _campaign_log_root(context: RunContext) -> Path:
    return context.logs_dir / "campaign"


def _campaign_state_path(context: RunContext) -> Path:
    return _campaign_log_root(context) / "campaign_state.json"


def _campaign_jsonl_path(context: RunContext) -> Path:
    return _campaign_log_root(context) / "campaign.jsonl"


def _append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _next_iteration(status: dict) -> int:
    latest = status.get("latest_iteration")
    if latest is None:
        return 0
    return int(latest) + 1


def _best_line(status: dict) -> str:
    best_iter = status.get("best_iteration")
    best_metric = status.get("best_metric")
    if best_iter is None or best_metric is None:
        return "No best kept iteration is recorded yet."
    return f"Best kept iteration so far: {best_iter} with score {best_metric}."


def _status_line(status: dict) -> str:
    return (
        f"Completed mutations: {status['completed_mutations']} / {status['target_iterations']}. "
        f"Next required iteration: {_next_iteration(status)}. Remaining mutations: {status['remaining_mutations']}."
    )


def _build_prompt(context: RunContext, status: dict) -> str:
    agent_prompt_path = context.root_dir / "agent_prompt.md"
    plan_path = context.root_dir / "plan.md"
    agent_prompt = agent_prompt_path.read_text().strip() if agent_prompt_path.exists() else ""
    plan_text = plan_path.read_text().strip() if plan_path.exists() else ""
    source_file = context.manifest["source_file"]
    run_dir = context.root_dir
    description = context.manifest.get("display_name", context.manifest.get("benchmark_value", run_dir.name))

    return f"""You are resuming an existing GSOpt campaign in `{run_dir}` for `{description}`.

Current campaign state:
- {_status_line(status)}
- {_best_line(status)}
- Live editable file: `{source_file}`

Requirements:
- Continue from the next required iteration and keep going until the target mutation count is reached.
- One outer iteration = exactly one explicit code mutation, then exactly one scored evaluation.
- Use `python3 run_eval.py -- uv run python evaluate.py --description "<technical mutation summary>"` for every scored iteration.
- Make every `--description` a short technical summary that explicitly names the mutation(s) you just made.
- Read the result of each scored evaluation before choosing the next mutation.
- If the last result is `discard` or `crash`, restore the best kept state before continuing.
- Do not stop after one iteration just because you made progress.
- Do not say you can continue later. Continue now until the run is complete or you hit a real blocker.
- Do not write internal parameter-sweep loops, hidden search drivers, or scripted batches of future mutations inside the benchmark file.
- Leave `{source_file}` in the best valid state archived so far whenever you exit.

Key run instructions:
{agent_prompt}

Plan summary:
{plan_text}
"""


def _launch_prefix(run_dir: Path, context: RunContext) -> list[str]:
    repo_root = context.repo_root
    prefix = []
    if repo_root != run_dir:
        prefix.extend(["--add-dir", str(repo_root)])
    return prefix


def _build_launch_spec(
    context: RunContext,
    agent: str,
    prompt: str,
    model: str | None,
    search: bool,
    agent_args: list[str],
    log_prefix: Path,
) -> LaunchSpec:
    run_dir = context.root_dir
    if agent == "codex":
        cmd = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(run_dir),
            "-o",
            str(log_prefix.with_suffix(".last_message.txt")),
            *_launch_prefix(run_dir, context),
        ]
        if model:
            cmd.extend(["-m", model])
        # Current Codex CLI exposes live search on the interactive command but
        # not on `codex exec`, so ignore the run-level search flag here rather
        # than generating repeated hard failures.
        cmd.extend(agent_args)
        cmd.append(prompt)
        return LaunchSpec(name="codex", command=cmd)

    if agent == "claude":
        cmd = [
            "claude",
            "-p",
            "--permission-mode",
            "bypassPermissions",
            *_launch_prefix(run_dir, context),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(agent_args)
        cmd.append(prompt)
        return LaunchSpec(name="claude", command=cmd)

    raise ValueError(f"unsupported agent {agent!r}")


def _write_campaign_state(context: RunContext, payload: dict):
    write_json(_campaign_state_path(context), payload)


def run_campaign(
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
    if not context.is_run:
        raise SystemExit("campaign mode requires a gsopt run directory; scaffold a run first")

    status = collect_status(context, write=True)
    if status["done"]:
        print(json.dumps({"done": True, "run_dir": str(context.root_dir), "status": status}, indent=2))
        return 0

    launches = 0
    consecutive_no_progress = 0
    log_root = _campaign_log_root(context)
    log_root.mkdir(parents=True, exist_ok=True)

    while launches < max_launches:
        status_before = collect_status(context, write=True)
        if status_before["done"]:
            break

        prompt = _build_prompt(context, status_before)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        launch_index = launches + 1
        log_prefix = log_root / f"launch_{launch_index:04d}_{stamp}"
        log_prefix.with_suffix(".prompt.txt").write_text(prompt)

        spec = _build_launch_spec(context, agent, prompt, model, search, agent_args, log_prefix)
        launch_event = {
            "timestamp": iso_now(),
            "type": "launch",
            "launch": launch_index,
            "agent": agent,
            "model": model,
            "cwd": str(context.root_dir),
            "command": spec.command,
            "status_before": {
                "completed_mutations": status_before["completed_mutations"],
                "latest_iteration": status_before.get("latest_iteration"),
                "remaining_mutations": status_before["remaining_mutations"],
            },
        }
        _append_jsonl(_campaign_jsonl_path(context), launch_event)
        _write_campaign_state(
            context,
            {
                "timestamp": iso_now(),
                "status": "launching",
                "launch": launch_index,
                "agent": agent,
                "model": model,
                "completed_mutations": status_before["completed_mutations"],
                "remaining_mutations": status_before["remaining_mutations"],
            },
        )

        if dry_run:
            print(json.dumps({"agent": agent, "command": spec.command, "prompt": prompt}, indent=2))
            return 0

        proc = subprocess.run(spec.command, cwd=context.root_dir, capture_output=True, text=True, check=False)
        launches += 1
        log_prefix.with_suffix(".stdout.txt").write_text(proc.stdout)
        log_prefix.with_suffix(".stderr.txt").write_text(proc.stderr)

        status_after = collect_status(context, write=True)
        completed_before = int(status_before["completed_mutations"])
        completed_after = int(status_after["completed_mutations"])
        progressed = completed_after > completed_before
        launch_summary = {
            "timestamp": iso_now(),
            "type": "launch_result",
            "launch": launch_index,
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
        _append_jsonl(_campaign_jsonl_path(context), launch_summary)
        _write_campaign_state(
            context,
            {
                "timestamp": iso_now(),
                "status": "waiting" if not status_after["done"] else "done",
                "launch": launch_index,
                "agent": agent,
                "model": model,
                "completed_mutations": completed_after,
                "remaining_mutations": status_after["remaining_mutations"],
                "last_returncode": proc.returncode,
            },
        )
        print(json.dumps(launch_summary), flush=True)

        if status_after["done"]:
            break

        if progressed:
            consecutive_no_progress = 0
        else:
            consecutive_no_progress += 1
            if consecutive_no_progress >= stall_launches:
                failure = {
                    "timestamp": iso_now(),
                    "status": "stalled",
                    "reason": "no_progress",
                    "launches": launches,
                    "completed_mutations": completed_after,
                }
                _append_jsonl(_campaign_jsonl_path(context), failure)
                _write_campaign_state(context, failure)
                print(json.dumps(failure), file=sys.stderr)
                return 1

        time.sleep(sleep_seconds)

    final_status = collect_status(context, write=True)
    done = bool(final_status["done"])
    summary = {
        "done": done,
        "run_dir": str(context.root_dir),
        "launches": launches,
        "completed_mutations": final_status["completed_mutations"],
        "target_iterations": final_status["target_iterations"],
        "latest_iteration": final_status.get("latest_iteration"),
    }
    _append_jsonl(_campaign_jsonl_path(context), {"timestamp": iso_now(), "type": "summary", **summary})
    _write_campaign_state(context, {"timestamp": iso_now(), "status": "done" if done else "incomplete", **summary})
    print(json.dumps(summary, indent=2))
    return 0 if done else 1
