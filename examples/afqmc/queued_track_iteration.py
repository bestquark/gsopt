from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from model_registry import ACTIVE_MOLECULES

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
QUEUE_ROOT = SCRIPT_DIR / "eval_queue"
SLOTS_DIR = QUEUE_ROOT / "slots"
REQUESTS_DIR = QUEUE_ROOT / "requests"
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_AFQMC_SNAPSHOT_ROOT", SCRIPT_DIR / "snapshots"))


def molecule_slug(name: str) -> str:
    return name.lower().replace("+", "_plus")


def next_iteration(molecule: str) -> int:
    root = SNAPSHOT_ROOT / molecule_slug(molecule)
    if not root.exists():
        return 0
    values = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            values.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(values, default=-1) + 1


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def clear_stale_dir(path: Path):
    metadata = read_json(path / "metadata.json")
    if metadata is None:
        shutil.rmtree(path, ignore_errors=True)
        return
    pid = int(metadata.get("pid", -1))
    if not pid_is_alive(pid):
        shutil.rmtree(path, ignore_errors=True)


def register_request(molecule: str, script: str, description: str, requested_iteration: int | None) -> Path:
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    request_id = f"req_{time.time_ns():020d}_{os.getpid()}"
    request_dir = REQUESTS_DIR / request_id
    request_dir.mkdir()
    metadata = {
        "pid": os.getpid(),
        "molecule": molecule,
        "script": script,
        "start_time": time.time(),
        "request_id": request_id,
        "description": description,
        "iteration": requested_iteration,
    }
    (request_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return request_dir


def active_requests() -> list[Path]:
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    for child in list(REQUESTS_DIR.iterdir()):
        if child.is_dir():
            clear_stale_dir(child)
    return sorted([child for child in REQUESTS_DIR.iterdir() if child.is_dir()], key=lambda path: path.name)


def request_rank(request_dir: Path) -> int | None:
    for idx, candidate in enumerate(active_requests()):
        if candidate == request_dir:
            return idx
    return None


def try_acquire_slot(
    max_parallel: int,
    molecule: str,
    script: str,
    request_id: str,
    description: str,
    requested_iteration: int | None,
) -> tuple[int, Path] | None:
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for idx in range(max_parallel):
        slot_dir = SLOTS_DIR / f"slot_{idx:02d}"
        if slot_dir.exists():
            clear_stale_dir(slot_dir)
        try:
            slot_dir.mkdir()
        except FileExistsError:
            continue
        metadata = {
            "pid": os.getpid(),
            "molecule": molecule,
            "script": script,
            "start_time": time.time(),
            "request_id": request_id,
            "description": description,
            "iteration": requested_iteration,
        }
        (slot_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return idx, slot_dir
    return None


def acquire_slot(
    max_parallel: int,
    molecule: str,
    script: str,
    poll_seconds: float,
    description: str,
    requested_iteration: int | None,
) -> tuple[int, Path, Path, float, int]:
    request_dir = register_request(molecule, script, description, requested_iteration)
    wait_start = time.perf_counter()
    while True:
        rank = request_rank(request_dir)
        if rank is None:
            request_dir = register_request(molecule, script, description, requested_iteration)
            rank = request_rank(request_dir)
        if rank is not None and rank < max_parallel:
            acquired = try_acquire_slot(
                max_parallel,
                molecule,
                script,
                request_dir.name,
                description,
                requested_iteration,
            )
            if acquired is not None:
                idx, slot_dir = acquired
                return idx, slot_dir, request_dir, time.perf_counter() - wait_start, rank
        time.sleep(poll_seconds)


def run_tracked_iteration(script: str, molecule: str, wall_seconds: float, iteration: int | None, description: str) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "track_iteration.py"),
        "--script",
        script,
        "--molecule",
        molecule,
        "--wall-seconds",
        str(wall_seconds),
        "--description",
        description,
    ]
    if iteration is not None:
        cmd.extend(["--iteration", str(iteration)])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "queued tracked iteration failed")
    return json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser(description="Queue-limited wrapper around AFQMC track_iteration.py.")
    parser.add_argument("--script", required=True, help="Path to the molecule-specific initial_script.py file.")
    parser.add_argument("--molecule", required=True, choices=ACTIVE_MOLECULES)
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--description", default="", help="Short note describing the code mutation.")
    args = parser.parse_args()

    slot_dir = None
    request_dir = None
    try:
        requested_iteration = args.iteration if args.iteration is not None else next_iteration(args.molecule)
        slot_idx, slot_dir, request_dir, wait_seconds, queue_rank = acquire_slot(
            max_parallel=args.max_parallel,
            molecule=args.molecule,
            script=args.script,
            poll_seconds=args.poll_seconds,
            description=args.description,
            requested_iteration=requested_iteration,
        )
        result = run_tracked_iteration(
            script=args.script,
            molecule=args.molecule,
            wall_seconds=args.wall_seconds,
            iteration=args.iteration,
            description=args.description,
        )
        result["queue_slot"] = slot_idx
        result["queue_wait_seconds"] = wait_seconds
        result["queue_max_parallel"] = args.max_parallel
        result["queue_rank_at_start"] = queue_rank
        result["queue_request_id"] = request_dir.name if request_dir is not None else None
        print(json.dumps(result, indent=2))
    finally:
        if slot_dir is not None and slot_dir.exists():
            shutil.rmtree(slot_dir, ignore_errors=True)
        if request_dir is not None and request_dir.exists():
            shutil.rmtree(request_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
