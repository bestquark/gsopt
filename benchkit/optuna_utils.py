from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

THREAD_BUDGET_ENV = {
    "OMP_NUM_THREADS": "10",
    "OPENBLAS_NUM_THREADS": "10",
    "MKL_NUM_THREADS": "10",
    "VECLIB_MAXIMUM_THREADS": "10",
    "NUMEXPR_NUM_THREADS": "10",
    "BLIS_NUM_THREADS": "10",
}


def slugify(name: str) -> str:
    return name.lower().replace("+", "_plus")


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def resolve_repo_path(repo_root: Path, path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_python_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def study_root(lane_dir: Path, label: str) -> Path:
    return lane_dir / "optuna" / slugify(label)


def source_archive_dir(lane_dir: Path, label: str) -> Path:
    return lane_dir / f"optuna_source_{slugify(label)}"


def prepare_frozen_source(
    lane_dir: Path,
    label: str,
    source_file: Path,
    reset: bool = False,
    archive_root: Path | None = None,
) -> tuple[Path, str, str]:
    frozen_dir = source_archive_dir(lane_dir, label) if archive_root is None else archive_root / "frozen_source"
    frozen_file = frozen_dir / source_file.name
    live_bytes = source_file.read_bytes()
    live_hash = sha256_bytes(live_bytes)

    if reset and frozen_dir.exists():
        shutil.rmtree(frozen_dir)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    if not frozen_file.exists():
        frozen_file.write_bytes(live_bytes)
    frozen_hash = sha256_bytes(frozen_file.read_bytes())
    return frozen_file, live_hash, frozen_hash


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_metadata(path: Path) -> dict | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None


def _clear_stale_dir(path: Path):
    metadata = _read_metadata(path)
    if metadata is None:
        shutil.rmtree(path, ignore_errors=True)
        return
    pid = int(metadata.get("pid", -1))
    if not pid_is_alive(pid):
        shutil.rmtree(path, ignore_errors=True)


def register_request(
    queue_root: Path,
    label_key: str,
    label_value: str,
    source_script: str,
    description: str,
    requested_index: int | None,
) -> Path:
    request_root = queue_root / "requests"
    request_root.mkdir(parents=True, exist_ok=True)
    request_id = f"req_{time.time_ns():020d}_{os.getpid()}"
    request_dir = request_root / request_id
    request_dir.mkdir()
    metadata = {
        "pid": os.getpid(),
        label_key: label_value,
        "script": source_script,
        "start_time": time.time(),
        "request_id": request_id,
        "description": description,
        "index": requested_index,
    }
    write_json(request_dir / "metadata.json", metadata)
    return request_dir


def active_requests(queue_root: Path) -> list[Path]:
    request_root = queue_root / "requests"
    request_root.mkdir(parents=True, exist_ok=True)
    for child in list(request_root.iterdir()):
        if child.is_dir():
            _clear_stale_dir(child)
    return sorted((child for child in request_root.iterdir() if child.is_dir()), key=lambda path: path.name)


def request_rank(queue_root: Path, request_dir: Path) -> int | None:
    for idx, candidate in enumerate(active_requests(queue_root)):
        if candidate == request_dir:
            return idx
    return None


def try_acquire_slot(
    queue_root: Path,
    max_parallel: int,
    label_key: str,
    label_value: str,
    source_script: str,
    request_id: str,
    description: str,
    requested_index: int | None,
) -> tuple[int, Path] | None:
    slots_root = queue_root / "slots"
    slots_root.mkdir(parents=True, exist_ok=True)
    for idx in range(max_parallel):
        slot_dir = slots_root / f"slot_{idx:02d}"
        if slot_dir.exists():
            _clear_stale_dir(slot_dir)
        try:
            slot_dir.mkdir()
        except FileExistsError:
            continue
        metadata = {
            "pid": os.getpid(),
            label_key: label_value,
            "script": source_script,
            "start_time": time.time(),
            "request_id": request_id,
            "description": description,
            "index": requested_index,
        }
        write_json(slot_dir / "metadata.json", metadata)
        return idx, slot_dir
    return None


def acquire_slot(
    queue_root: Path,
    max_parallel: int,
    poll_seconds: float,
    label_key: str,
    label_value: str,
    source_script: str,
    description: str,
    requested_index: int | None,
) -> tuple[int, Path, Path, float, int]:
    request_dir = register_request(
        queue_root=queue_root,
        label_key=label_key,
        label_value=label_value,
        source_script=source_script,
        description=description,
        requested_index=requested_index,
    )
    wait_start = time.perf_counter()
    while True:
        rank = request_rank(queue_root, request_dir)
        if rank is None:
            request_dir = register_request(
                queue_root=queue_root,
                label_key=label_key,
                label_value=label_value,
                source_script=source_script,
                description=description,
                requested_index=requested_index,
            )
            rank = request_rank(queue_root, request_dir)
        if rank is not None and rank < max_parallel:
            acquired = try_acquire_slot(
                queue_root=queue_root,
                max_parallel=max_parallel,
                label_key=label_key,
                label_value=label_value,
                source_script=source_script,
                request_id=request_dir.name,
                description=description,
                requested_index=requested_index,
            )
            if acquired is not None:
                slot_idx, slot_dir = acquired
                return slot_idx, slot_dir, request_dir, time.perf_counter() - wait_start, rank
        time.sleep(poll_seconds)
