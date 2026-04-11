from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path


def _locate_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    raise FileNotFoundError(f"could not locate repo root from {start}")


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _normalize_archive_args(args: list[str], benchmark_dir: Path) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--archive-root" and idx + 1 < len(args):
            value = Path(args[idx + 1])
            if not value.is_absolute():
                value = (benchmark_dir / value).resolve()
            normalized.extend([arg, str(value)])
            idx += 2
            continue
        if arg.startswith("--archive-root="):
            value = Path(arg.split("=", 1)[1])
            if not value.is_absolute():
                value = (benchmark_dir / value).resolve()
            normalized.append(f"--archive-root={value}")
            idx += 1
            continue
        normalized.append(arg)
        idx += 1
    if not _has_option(normalized, "--archive-root"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized.extend(["--archive-root", str((benchmark_dir / f"optuna_run_{stamp}").resolve())])
    return normalized


def exec_lane_optuna(
    *,
    wrapper_file: str,
    lane_script: str,
    source_filename: str,
    label_flag: str,
    label_value: str,
) -> int:
    wrapper_path = Path(wrapper_file).resolve()
    benchmark_dir = wrapper_path.parent
    repo_root = _locate_repo_root(wrapper_path)
    lane_script_path = (repo_root / lane_script).resolve()

    for path in (repo_root, lane_script_path.parent):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    module_name = f"optuna_wrapper_{lane_script_path.parent.name}_{benchmark_dir.name}"
    module = _load_module(lane_script_path, module_name)

    cli_args = _normalize_archive_args(sys.argv[1:], benchmark_dir)
    source_path = (benchmark_dir / source_filename).resolve()
    argv = [sys.argv[0], "--script", str(source_path), label_flag, label_value, *cli_args]
    original_argv = sys.argv
    try:
        sys.argv = argv
        return int(module.main())
    finally:
        sys.argv = original_argv
