from __future__ import annotations

import json
import time
from pathlib import Path

from .runtime import collect_status, locate_context


def run_watchdog(run_dir: Path, poll_seconds: float, stall_seconds: float) -> int:
    context = locate_context(run_dir)
    log_path = context.logs_dir / "watchdog.jsonl"
    last_progress_time = time.time()
    last_completed = None
    while True:
        status = collect_status(context, write=True)
        completed = status["completed_mutations"]
        event = {
            "timestamp": status["timestamp"],
            "completed_mutations": completed,
            "remaining_mutations": status["remaining_mutations"],
            "done": status["done"],
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        if status["done"]:
            return 0
        if last_completed is None or completed > last_completed:
            last_completed = completed
            last_progress_time = time.time()
        elif time.time() - last_progress_time > stall_seconds:
            return 1
        time.sleep(poll_seconds)
