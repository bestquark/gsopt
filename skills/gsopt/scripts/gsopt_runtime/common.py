from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAMES = (".gsopt.json",)
DEFAULT_EVALUATOR_CANDIDATES = ("evaluate.py", "evaluator.py", "eval.py")
DEFAULT_SOURCE_CANDIDATES = (
    "initial_script.py",
    "simple_vqe.py",
    "initial_program.py",
    "simple_dmrg.py",
    "simple_gibbs_mcmc.py",
    "program.py",
    "method.py",
    "main.py",
)
RUNTIME_HELPER_FILENAMES = frozenset(
    DEFAULT_EVALUATOR_CANDIDATES
    + (
        "optuna_baseline.py",
        "status.py",
        "plot.py",
        "restore_best.py",
        "run_eval.py",
        "watchdog.py",
        "campaign.py",
    )
)


def find_skill_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "SKILL.md").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError(f"could not locate gsopt skill root from {current}")


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skills" / "gsopt" / "SKILL.md").exists():
            return candidate
    skill_root = find_skill_root()
    for candidate in (skill_root.parent, *skill_root.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skills" / "gsopt" / "SKILL.md").exists():
            return candidate
    return skill_root.parent


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


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def infer_evaluator_file(benchmark_root: Path) -> Path:
    for name in DEFAULT_EVALUATOR_CANDIDATES:
        candidate = benchmark_root / name
        if candidate.exists():
            return candidate
    expected = ", ".join(DEFAULT_EVALUATOR_CANDIDATES)
    raise FileNotFoundError(
        f"no evaluator file found in {benchmark_root}; expected one of {expected}. "
        "Create an evaluator that prints JSON with a scalar `score`, or rerun gsopt with "
        "`--evaluator <path>`."
    )


def infer_source_file(benchmark_root: Path, evaluator_file: Path | None = None) -> Path:
    evaluator_path = None if evaluator_file is None else evaluator_file.resolve()
    for name in DEFAULT_SOURCE_CANDIDATES:
        candidate = benchmark_root / name
        if candidate.exists() and candidate.resolve() != evaluator_path:
            return candidate
    candidates = []
    for child in sorted(benchmark_root.iterdir()):
        if not child.is_file() or child.suffix != ".py":
            continue
        if child.name in RUNTIME_HELPER_FILENAMES:
            continue
        if evaluator_path is not None and child.resolve() == evaluator_path:
            continue
        candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        visible = ", ".join(path.name for path in candidates[:8])
        if len(candidates) > 8:
            visible += ", ..."
        raise FileNotFoundError(
            f"could not infer the editable source file in {benchmark_root}; found multiple Python files "
            f"({visible}). Rerun gsopt with `--source <path>`."
        )
    raise FileNotFoundError(
        f"could not infer the editable source file in {benchmark_root}; expected one of "
        f"{', '.join(DEFAULT_SOURCE_CANDIDATES)}. Add one such file or rerun gsopt with `--source <path>`."
    )


def synthesize_manifest(
    benchmark_root: Path,
    source_file: Path | None = None,
    evaluator_file: Path | None = None,
) -> dict[str, Any]:
    evaluator = evaluator_file or infer_evaluator_file(benchmark_root)
    source = source_file or infer_source_file(benchmark_root, evaluator_file=evaluator)
    return {
        "version": 1,
        "lane": "generic",
        "example_key": benchmark_root.name,
        "benchmark_arg": "benchmark",
        "benchmark_value": benchmark_root.name,
        "display_name": benchmark_root.name,
        "source_file": _relative_to_root(source, benchmark_root),
        "evaluator_file": _relative_to_root(evaluator, benchmark_root),
        "source_template": str(source.resolve()),
        "queue_script": None,
        "restore_script": None,
        "plot_script": None,
        "optuna_script": None,
        "snapshot_env": None,
        "fig_dir_env": None,
        "run_root_env": None,
        "optuna_root_env": None,
        "default_iterations": 100,
        "default_wall_seconds": 20.0,
        "objective_metric": "score",
        "objective_text": "Lower the scalar score returned by the evaluator under the benchmark's stated constraints.",
        "support_files": [],
        "benchmark_storage_name": benchmark_root.name,
    }
