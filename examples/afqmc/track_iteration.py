from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from model_registry import ACTIVE_MOLECULES

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_AFQMC_SNAPSHOT_ROOT", SCRIPT_DIR / "snapshots"))
THREAD_BUDGET_ENV = {
    "OMP_NUM_THREADS": "10",
    "OPENBLAS_NUM_THREADS": "10",
    "MKL_NUM_THREADS": "10",
    "VECLIB_MAXIMUM_THREADS": "10",
    "NUMEXPR_NUM_THREADS": "10",
    "BLIS_NUM_THREADS": "10",
}


def molecule_slug(name: str) -> str:
    return name.lower().replace("+", "_plus")


def snapshot_dir(molecule: str, iteration: int) -> Path:
    return SNAPSHOT_ROOT / molecule_slug(molecule) / f"iter_{iteration:04d}"


def molecule_root(molecule: str) -> Path:
    return SNAPSHOT_ROOT / molecule_slug(molecule)


def results_path(molecule: str) -> Path:
    return molecule_root(molecule) / "results.tsv"


def next_iteration(molecule: str) -> int:
    root = molecule_root(molecule)
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


def resolve_script(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_source_file(source_file: Path) -> tuple[Path, tempfile.TemporaryDirectory[str], bytes, str]:
    source_bytes = source_file.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    staging_dir = tempfile.TemporaryDirectory(prefix="afqmc_track_", dir=SCRIPT_DIR)
    frozen_file = Path(staging_dir.name) / source_file.name
    frozen_file.write_bytes(source_bytes)
    return frozen_file, staging_dir, source_bytes, source_hash


def run_evaluation(source_file: Path, wall_seconds: float) -> tuple[dict | None, str, str, str | None]:
    env = os.environ.copy()
    env.update(THREAD_BUDGET_ENV)
    proc = subprocess.run(
        [sys.executable, str(source_file), "--wall-seconds", str(wall_seconds)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return None, stdout, stderr, stderr or stdout or "evaluation failed"
    if not stdout:
        return None, stdout, stderr, "evaluation produced no stdout"
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, stdout, stderr, f"evaluation stdout was not valid JSON: {exc}"
    return payload, stdout, stderr, None


def git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def previous_snapshot_file(molecule: str, iteration: int) -> Path | None:
    for idx in range(iteration - 1, -1, -1):
        candidate = snapshot_source_file(snapshot_dir(molecule, idx))
        if candidate.exists():
            return candidate
    return None


def snapshot_source_file(root: Path) -> Path:
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
            archived_name = payload.get("archived_source_name")
            if archived_name:
                candidate = root / str(archived_name)
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass
    for name in ("initial_script.py", "simple_afqmc.py"):
        candidate = root / name
        if candidate.exists():
            return candidate
    python_files = sorted(root.glob("*.py"))
    if python_files:
        return python_files[0]
    return root / "initial_script.py"


def write_diff(previous_file: Path | None, current_file: Path, output_file: Path):
    if previous_file is None:
        output_file.write_text("")
        return
    prev_lines = previous_file.read_text().splitlines(keepends=True)
    curr_lines = current_file.read_text().splitlines(keepends=True)
    diff = difflib.unified_diff(prev_lines, curr_lines, fromfile=str(previous_file), tofile=str(current_file))
    output_file.write_text("".join(diff))


def append_summary(summary_path: Path, row: dict):
    if not summary_path.exists():
        summary_path.write_text(
            "iteration\tfinal_energy\treference_energy\tfinal_error\tabs_final_error\twall_seconds\tsnapshot_dir\n"
        )
    with summary_path.open("a") as handle:
        handle.write(
            f"{row['iteration']}\t{row['final_energy']:.12f}\t{row['reference_energy']:.12f}\t"
            f"{row['final_error']:.12e}\t{row['abs_final_error']:.12e}\t"
            f"{row['wall_seconds']:.4f}\t{row['snapshot_dir']}\n"
        )


def exact_abs_error_from_snapshot(snapshot_dir_text: str, fallback: str) -> float | None:
    snapshot_dir = Path(snapshot_dir_text)
    result_path = snapshot_dir / "result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text())
            return abs(float(payload["final_error"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if fallback == "":
        return None
    try:
        return float(fallback)
    except ValueError:
        return None


def best_abs_final_error(molecule: str) -> tuple[int | None, float | None]:
    path = results_path(molecule)
    if not path.exists():
        return None, None
    best_iteration = None
    best_value = None
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            iteration_text = row.get("iteration", "")
            metric = exact_abs_error_from_snapshot(row.get("snapshot_dir", ""), row.get("abs_final_error", ""))
            if metric is None or iteration_text == "":
                continue
            try:
                iteration = int(iteration_text)
            except ValueError:
                continue
            if best_value is None or metric < best_value:
                best_iteration = iteration
                best_value = metric
    return best_iteration, best_value


def append_results(results_file: Path, row: dict):
    if not results_file.exists():
        results_file.write_text(
            "iteration\tabs_final_error\tfinal_energy\treference_energy\tfinal_error\twall_seconds\tstatus\tdescription\tsnapshot_dir\n"
        )
    with results_file.open("a") as handle:
        abs_final_error = "" if row["abs_final_error"] is None else f"{row['abs_final_error']:.12e}"
        final_energy = "" if row["final_energy"] is None else f"{row['final_energy']:.12f}"
        reference_energy = "" if row["reference_energy"] is None else f"{row['reference_energy']:.12f}"
        final_error = "" if row["final_error"] is None else f"{row['final_error']:.12e}"
        description = row["description"].replace("\t", " ").strip()
        handle.write(
            f"{row['iteration']}\t{abs_final_error}\t{final_energy}\t{reference_energy}\t{final_error}\t"
            f"{row['wall_seconds']:.4f}\t{row['status']}\t{description}\t{row['snapshot_dir']}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Run and archive one editable AFQMC iteration.")
    parser.add_argument("--molecule", required=True, choices=ACTIVE_MOLECULES)
    parser.add_argument("--script", required=True, help="Path to the molecule-specific initial_script.py file.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--description", default="", help="Short note describing the code mutation.")
    args = parser.parse_args()

    source_file = resolve_script(args.script)
    if not source_file.exists():
        raise SystemExit(f"missing source file: {source_file}")

    iteration = args.iteration if args.iteration is not None else next_iteration(args.molecule)
    out_dir = snapshot_dir(args.molecule, iteration)
    if out_dir.exists():
        raise SystemExit(f"snapshot directory already exists: {out_dir}")

    previous_best_iteration, previous_best_abs_error = best_abs_final_error(args.molecule)
    frozen_file, staging_dir, source_bytes, source_hash_before = freeze_source_file(source_file)
    try:
        result, stdout, stderr, error = run_evaluation(frozen_file, args.wall_seconds)
    finally:
        staging_dir.cleanup()

    live_source_exists_after = source_file.exists()
    live_source_bytes_after = source_file.read_bytes() if live_source_exists_after else b""
    live_source_hash_after = sha256_bytes(live_source_bytes_after) if live_source_exists_after else None
    live_source_changed_during_run = (not live_source_exists_after) or (live_source_bytes_after != source_bytes)

    out_dir.mkdir(parents=True, exist_ok=False)
    archived_source = out_dir / source_file.name
    archived_source.write_bytes(source_bytes)
    if result is not None:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    if stdout:
        (out_dir / "stdout.txt").write_text(stdout + "\n")
    if stderr:
        (out_dir / "stderr.txt").write_text(stderr + "\n")
    if error is not None:
        (out_dir / "error.txt").write_text(error + "\n")
    if live_source_changed_during_run:
        (out_dir / "live_source_changed.txt").write_text(
            "The live source file changed while this archived iteration was running.\n"
        )

    metadata = {
        "iteration": iteration,
        "molecule": args.molecule,
        "wall_seconds": args.wall_seconds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "source_file": str(source_file),
        "archived_source_name": source_file.name,
        "description": args.description,
        "source_sha256_before_run": source_hash_before,
        "live_source_sha256_after_run": live_source_hash_after,
        "live_source_changed_during_run": live_source_changed_during_run,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    prev_file = previous_snapshot_file(args.molecule, iteration)
    write_diff(prev_file, archived_source, out_dir / "changes.diff")

    status = "crash"
    abs_final_error = None
    final_energy = None
    ref_energy = None
    final_error = None
    wall_seconds = args.wall_seconds
    if result is not None:
        final_energy = result["final_energy"]
        ref_energy = result["reference_energy"]
        final_error = result["final_error"]
        abs_final_error = abs(final_error)
        wall_seconds = result["wall_seconds"]
        status = "keep" if previous_best_abs_error is None or abs_final_error < previous_best_abs_error else "discard"
        append_summary(
            molecule_root(args.molecule) / "summary.tsv",
            {
                "iteration": iteration,
                "final_energy": final_energy,
                "reference_energy": ref_energy,
                "final_error": final_error,
                "abs_final_error": abs_final_error,
                "wall_seconds": wall_seconds,
                "snapshot_dir": str(out_dir),
            },
        )

    append_results(
        results_path(args.molecule),
        {
            "iteration": iteration,
            "abs_final_error": abs_final_error,
            "final_energy": final_energy,
            "reference_energy": ref_energy,
            "final_error": final_error,
            "wall_seconds": wall_seconds,
            "status": status,
            "description": args.description,
            "snapshot_dir": out_dir,
        },
    )

    best_iteration, best_abs_error = best_abs_final_error(args.molecule)
    print(
        json.dumps(
            {
                "iteration": iteration,
                "status": status,
                "snapshot_dir": str(out_dir),
                "final_energy": final_energy,
                "reference_energy": ref_energy,
                "final_error": final_error,
                "abs_final_error": abs_final_error,
                "wall_seconds": wall_seconds,
                "description": args.description,
                "best_iteration_so_far": best_iteration,
                "best_abs_final_error_so_far": best_abs_error,
                "previous_best_iteration": previous_best_iteration,
                "previous_best_abs_final_error": previous_best_abs_error,
                "live_source_changed_during_run": live_source_changed_during_run,
                "error": error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
