from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAMES = (".gsopt.json", ".energyopt.json", "gsopt.json", "energyopt.json")


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    raise FileNotFoundError(f"could not locate repo root from {current}")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def benchmark_slug(value: str) -> str:
    return value.lower().replace("+", "_plus")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text)


def find_manifest_path(directory: Path) -> Path | None:
    for name in MANIFEST_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def copy_tree_contents(source: Path, destination: Path, exclude: tuple[str, ...] = ("runs", "__pycache__")):
    destination.mkdir(parents=True, exist_ok=True)
    ignore = set(exclude)
    for child in source.iterdir():
        if child.name in ignore:
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(child, target)


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())
