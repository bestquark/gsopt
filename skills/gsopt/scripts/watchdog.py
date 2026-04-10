#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


HEARTBEAT = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/launcher/heartbeat.json")
PID_FILE = Path(sys.argv[2] if len(sys.argv) > 2 else "logs/launcher/worker.pid")
TIMEOUT_S = int(sys.argv[3] if len(sys.argv) > 3 else 180)
POLL_S = float(sys.argv[4] if len(sys.argv) > 4 else 15.0)


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


while True:
    time.sleep(POLL_S)
    pid = read_pid()
    if pid is None or not process_alive(pid) or not HEARTBEAT.exists():
        continue
    age = time.time() - HEARTBEAT.stat().st_mtime
    if age <= TIMEOUT_S:
        continue
    print(f"[gsopt-watchdog] stale heartbeat ({age:.1f}s), terminating pid={pid}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
