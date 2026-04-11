from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from model_registry import ACTIVE_MODELS
from reference_energies import reference_energy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SNAPSHOT_ROOT = Path(os.environ.get("AUTORESEARCH_TN_SNAPSHOT_ROOT", SCRIPT_DIR / "snapshots"))
THREAD_BUDGET_ENV = {
    "OMP_NUM_THREADS": "10",
    "OPENBLAS_NUM_THREADS": "10",
    "MKL_NUM_THREADS": "10",
    "VECLIB_MAXIMUM_THREADS": "10",
    "NUMEXPR_NUM_THREADS": "10",
    "BLIS_NUM_THREADS": "10",
}

SUPPORTED_MODELS = ACTIVE_MODELS


def snapshot_dir(model: str, iteration: int) -> Path:
    return SNAPSHOT_ROOT / model / f"iter_{iteration:04d}"


def model_root(model: str) -> Path:
    return SNAPSHOT_ROOT / model


def results_path(model: str) -> Path:
    return model_root(model) / "results.tsv"


def next_iteration(model: str) -> int:
    root = model_root(model)
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
    staging_dir = tempfile.TemporaryDirectory(prefix="tn_track_", dir=SCRIPT_DIR)
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


def previous_snapshot_file(model: str, iteration: int) -> Path | None:
    for idx in range(iteration - 1, -1, -1):
        candidate = snapshot_source_file(snapshot_dir(model, idx))
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
    for name in ("initial_script.py",):
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
            "iteration\tfinal_energy\treference_energy\texcess_energy\tenergy_per_site\treference_energy_per_site\t"
            "excess_energy_per_site\tenergy_drop\twall_seconds\tsnapshot_dir\n"
        )
    with summary_path.open("a") as handle:
        reference_value = "" if row["reference_energy"] is None else f"{row['reference_energy']:.12f}"
        excess_energy = "" if row["excess_energy"] is None else f"{row['excess_energy']:.6e}"
        reference_per_site = (
            "" if row["reference_energy_per_site"] is None else f"{row['reference_energy_per_site']:.12f}"
        )
        excess_per_site = "" if row["excess_energy_per_site"] is None else f"{row['excess_energy_per_site']:.6e}"
        handle.write(
            f"{row['iteration']}\t{row['final_energy']:.12f}\t{reference_value}\t"
            f"{excess_energy}\t{row['energy_per_site']:.12f}\t"
            f"{reference_per_site}\t{excess_per_site}\t"
            f"{row['energy_drop']:.6e}\t{row['wall_seconds']:.4f}\t{row['snapshot_dir']}\n"
        )


def best_final_energy(model: str) -> tuple[int | None, float | None]:
    path = results_path(model)
    if not path.exists():
        return None, None
    best_iteration = None
    best_value = None
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep" or not row.get("final_energy"):
                continue
            iteration = int(row["iteration"])
            value = float(row["final_energy"])
            if best_value is None or value < best_value:
                best_iteration = iteration
                best_value = value
    return best_iteration, best_value


def append_results(results_file: Path, row: dict):
    if not results_file.exists():
        results_file.write_text(
            "iteration\tfinal_energy\treference_energy\texcess_energy\tenergy_per_site\treference_energy_per_site\t"
            "excess_energy_per_site\tenergy_drop\twall_seconds\tstatus\tdescription\tsnapshot_dir\n"
        )
    with results_file.open("a") as handle:
        final_energy = "" if row["final_energy"] is None else f"{row['final_energy']:.12f}"
        reference_value = "" if row["reference_energy"] is None else f"{row['reference_energy']:.12f}"
        excess_energy = "" if row["excess_energy"] is None else f"{row['excess_energy']:.6e}"
        energy_per_site = "" if row["energy_per_site"] is None else f"{row['energy_per_site']:.12f}"
        reference_per_site = (
            "" if row["reference_energy_per_site"] is None else f"{row['reference_energy_per_site']:.12f}"
        )
        excess_per_site = "" if row["excess_energy_per_site"] is None else f"{row['excess_energy_per_site']:.6e}"
        energy_drop = "" if row["energy_drop"] is None else f"{row['energy_drop']:.6e}"
        description = row["description"].replace("\t", " ").strip()
        handle.write(
            f"{row['iteration']}\t{final_energy}\t{reference_value}\t{excess_energy}\t{energy_per_site}\t"
            f"{reference_per_site}\t{excess_per_site}\t{energy_drop}\t{row['wall_seconds']:.4f}\t"
            f"{row['status']}\t{description}\t{row['snapshot_dir']}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Run and archive one editable tensor-network iteration.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--script", required=True, help="Path to the model-specific initial_script.py file.")
    parser.add_argument("--wall-seconds", type=float, default=20.0)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--description", default="", help="Short note describing the code mutation.")
    args = parser.parse_args()

    source_file = resolve_script(args.script)
    if not source_file.exists():
        raise SystemExit(f"missing source file: {source_file}")

    iteration = args.iteration if args.iteration is not None else next_iteration(args.model)
    out_dir = snapshot_dir(args.model, iteration)
    if out_dir.exists():
        raise SystemExit(f"snapshot directory already exists: {out_dir}")

    previous_best_iteration, previous_best_energy = best_final_energy(args.model)
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
        "model": args.model,
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

    prev_file = previous_snapshot_file(args.model, iteration)
    write_diff(prev_file, archived_source, out_dir / "changes.diff")

    status = "crash"
    final_energy = None
    reference_value = reference_energy(args.model)
    excess_energy = None
    energy_per_site = None
    reference_per_site = None
    excess_per_site = None
    energy_drop = None
    wall_seconds = args.wall_seconds
    if result is not None:
        final_energy = result["final_energy"]
        energy_per_site = result["energy_per_site"]
        energy_drop = result["energy_drop"]
        wall_seconds = result["wall_seconds"]
        if reference_value is not None:
            excess_energy = final_energy - reference_value
            reference_per_site = reference_value / result["chain_length"]
            excess_per_site = energy_per_site - reference_per_site
        status = "keep" if previous_best_energy is None or final_energy < previous_best_energy else "discard"
        append_summary(
            model_root(args.model) / "summary.tsv",
            {
                "iteration": iteration,
                "final_energy": final_energy,
                "reference_energy": reference_value,
                "excess_energy": excess_energy,
                "energy_per_site": energy_per_site,
                "reference_energy_per_site": reference_per_site,
                "excess_energy_per_site": excess_per_site,
                "energy_drop": energy_drop,
                "wall_seconds": wall_seconds,
                "snapshot_dir": str(out_dir),
            },
        )

    append_results(
        results_path(args.model),
        {
            "iteration": iteration,
            "final_energy": final_energy,
            "reference_energy": reference_value,
            "excess_energy": excess_energy,
            "energy_per_site": energy_per_site,
            "reference_energy_per_site": reference_per_site,
            "excess_energy_per_site": excess_per_site,
            "energy_drop": energy_drop,
            "wall_seconds": wall_seconds,
            "status": status,
            "description": args.description,
            "snapshot_dir": out_dir,
        },
    )

    best_iteration, best_energy = best_final_energy(args.model)
    print(
        json.dumps(
            {
                "iteration": iteration,
                "status": status,
                "snapshot_dir": str(out_dir),
                "final_energy": final_energy,
                "reference_energy": reference_value,
                "excess_energy": excess_energy,
                "energy_per_site": energy_per_site,
                "reference_energy_per_site": reference_per_site,
                "excess_energy_per_site": excess_per_site,
                "energy_drop": energy_drop,
                "wall_seconds": wall_seconds,
                "description": args.description,
                "best_iteration_so_far": best_iteration,
                "best_final_energy_so_far": best_energy,
                "previous_best_iteration": previous_best_iteration,
                "previous_best_final_energy": previous_best_energy,
                "live_source_changed_during_run": live_source_changed_during_run,
                "error": error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
