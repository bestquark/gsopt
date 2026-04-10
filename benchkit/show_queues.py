from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE_QUEUE_ROOTS = {
    "vqe": ROOT / "examples" / "vqe" / "eval_queue",
    "dmrg": ROOT / "examples" / "dmrg" / "eval_queue",
    "tn": ROOT / "examples" / "tn" / "eval_queue",
    "afqmc": ROOT / "examples" / "afqmc" / "eval_queue",
}


@dataclass
class QueueEntry:
    lane: str
    request_id: str
    label: str
    pid: int | None
    start_time: float | None
    script: str
    description: str = ""
    iteration: int | None = None
    queue_rank: int | None = None
    slot_name: str | None = None

    @property
    def age_seconds(self) -> float | None:
        if self.start_time is None:
            return None
        return max(0.0, time.time() - self.start_time)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def clear_stale_dir(path: Path):
    metadata = read_json(path / "metadata.json")
    if metadata is None:
        shutil.rmtree(path, ignore_errors=True)
        return
    pid = int(metadata["pid"]) if "pid" in metadata else None
    if not pid_is_alive(pid):
        shutil.rmtree(path, ignore_errors=True)


def request_entries(lane: str, queue_root: Path) -> list[QueueEntry]:
    requests_dir = queue_root / "requests"
    if not requests_dir.exists():
        return []
    for request_dir in list(requests_dir.iterdir()):
        if request_dir.is_dir():
            clear_stale_dir(request_dir)
    entries: list[QueueEntry] = []
    for rank, request_dir in enumerate(sorted(child for child in requests_dir.iterdir() if child.is_dir())):
        metadata = read_json(request_dir / "metadata.json")
        if metadata is None:
            continue
        entries.append(
            QueueEntry(
                lane=lane,
                request_id=request_dir.name,
                label=metadata.get("molecule") or metadata.get("model") or "?",
                pid=int(metadata["pid"]) if "pid" in metadata else None,
                start_time=float(metadata["start_time"]) if "start_time" in metadata else None,
                script=Path(metadata.get("script", "")).name,
                description=str(metadata.get("description", "")).strip(),
                iteration=int(metadata["iteration"]) if metadata.get("iteration") is not None else None,
                queue_rank=rank,
            )
        )
    return entries


def slot_entries(lane: str, queue_root: Path) -> list[QueueEntry]:
    slots_dir = queue_root / "slots"
    if not slots_dir.exists():
        return []
    for slot_dir in list(slots_dir.iterdir()):
        if slot_dir.is_dir():
            clear_stale_dir(slot_dir)
    entries: list[QueueEntry] = []
    for slot_dir in sorted(child for child in slots_dir.iterdir() if child.is_dir()):
        metadata = read_json(slot_dir / "metadata.json")
        if metadata is None:
            continue
        entries.append(
            QueueEntry(
                lane=lane,
                request_id=metadata.get("request_id", slot_dir.name),
                label=metadata.get("molecule") or metadata.get("model") or "?",
                pid=int(metadata["pid"]) if "pid" in metadata else None,
                start_time=float(metadata["start_time"]) if "start_time" in metadata else None,
                script=Path(metadata.get("script", "")).name,
                description=str(metadata.get("description", "")).strip(),
                iteration=int(metadata["iteration"]) if metadata.get("iteration") is not None else None,
                slot_name=slot_dir.name,
            )
        )
    return entries


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 60:
        return f"{seconds:5.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes):2d}m{int(rem):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):2d}h{int(minutes):02d}m"


def render_lane(lane: str):
    queue_root = LANE_QUEUE_ROOTS[lane]
    request_map = {entry.request_id: entry for entry in request_entries(lane, queue_root)}
    slot_list = slot_entries(lane, queue_root)
    active_ids = {entry.request_id for entry in slot_list}

    print(f"{lane.upper()} queue")
    print(f"  root: {queue_root}")
    if slot_list:
        print("  active:")
        for entry in slot_list:
            iteration_text = "?" if entry.iteration is None else str(entry.iteration)
            description_suffix = f", {entry.description}" if entry.description else ""
            print(
                f"    {entry.slot_name}: {entry.label} iter {iteration_text} "
                f"(age {format_age(entry.age_seconds)}, pid {entry.pid}, {entry.script}{description_suffix})"
            )
    else:
        print("  active: none")

    waiting = [entry for request_id, entry in request_map.items() if request_id not in active_ids]
    if waiting:
        print("  waiting:")
        for entry in waiting:
            rank = "?" if entry.queue_rank is None else entry.queue_rank
            iteration_text = "?" if entry.iteration is None else str(entry.iteration)
            description_suffix = f", {entry.description}" if entry.description else ""
            print(
                f"    rank {rank}: {entry.label} iter {iteration_text} "
                f"(age {format_age(entry.age_seconds)}, pid {entry.pid}, {entry.script}{description_suffix})"
            )
    else:
        print("  waiting: none")


def render(lane: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print()
    lanes = list(LANE_QUEUE_ROOTS) if lane == "all" else [lane]
    for idx, item in enumerate(lanes):
        render_lane(item)
        if idx != len(lanes) - 1:
            print()


def main():
    parser = argparse.ArgumentParser(description="Show the live VQE/DMRG/TN/AFQMC queue state.")
    parser.add_argument("--lane", choices=["all", "vqe", "dmrg", "tn", "afqmc"], default="all")
    parser.add_argument("--follow", action="store_true", help="Refresh continuously.")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds for --follow.")
    args = parser.parse_args()

    if not args.follow:
        render(args.lane)
        return

    try:
        while True:
            print("\033[2J\033[H", end="")
            render(args.lane)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
