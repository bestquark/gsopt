from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from .common import find_manifest_path, find_repo_root, iso_now, write_json
from .registry import ExampleSpec, examples_by_benchmark_root, examples_by_source, load_examples

MANIFEST_NAME = ".gsopt.json"

WRAPPER_TEMPLATE = """from __future__ import annotations

import sys
from pathlib import Path


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError(f"could not locate repo root from {{start}}")


ROOT = _repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchkit.entrypoints import {entrypoint}


if __name__ == "__main__":
    raise SystemExit({entrypoint}())
"""


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _benchmark_root(repo_root: Path, example: ExampleSpec) -> Path:
    return (repo_root / example.source_template).resolve().parent


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _copy_source_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _benchmark_root_from_candidate(candidate: Path) -> Path:
    run_meta_path = candidate / "run.json"
    if run_meta_path.exists():
        return Path(json.loads(run_meta_path.read_text())["benchmark_root"]).resolve()
    return candidate.resolve()


def _plan_markdown(manifest: dict, run_meta: dict) -> str:
    source_file = manifest["source_file"]
    objective = manifest["objective_text"]
    evaluation_mode = run_meta["evaluation_mode"]
    max_parallel = run_meta["max_parallel"]
    queue_line = (
        "Serialized scoring: at most one scored evaluation runs at a time."
        if evaluation_mode == "serialized"
        else f"Parallel scoring is allowed, with up to `{max_parallel}` concurrent queued evaluations."
    )
    return f"""# GSOpt Run Plan

- Created: `{run_meta["created_at"]}`
- Target iterations: `{run_meta["target_iterations"]}`
- Benchmark: `{manifest["benchmark_value"]}` ({manifest["lane"]})
- File to evolve: `{source_file}`
- Objective: {objective}
- Evaluation mode: `{evaluation_mode}`
- Queue width: `{max_parallel}`

## Commands

Baseline and sequential scored iterations:

```bash
uv run gsopt run-eval -- uv run python evaluate.py --description "<one-line mutation summary>"
```

Restore the best kept iteration:

```bash
uv run python restore_best.py
```

Check progress:

```bash
uv run python status.py
```

Render local figures:

```bash
uv run python plot.py
```

## Search policy

- {queue_line}
- Archive the untouched baseline as iteration `0`.
- Then run exactly iterations `1` through `{run_meta["target_iterations"]}`.
- One outer iteration = one explicit code mutation plus one queued scored run.
- Read the last scored result before choosing the next mutation.
- If a scored run is `discard` or `crash`, restore the best kept iteration before continuing.
- Be really creative about reducing the energy: once tiny tolerance or seed tweaks plateau, prefer structural changes that alter the search geometry or ansatz quality.
- Do not pre-script batches of future iterations, seed sweeps, or offline probes outside the queued evaluator.
"""


def _agent_prompt(manifest: dict, run_meta: dict) -> str:
    extra = run_meta["additional_instructions"].strip()
    extra_block = f"\nAdditional user instructions:\n{extra}\n" if extra else ""
    evaluation_mode = run_meta["evaluation_mode"]
    max_parallel = run_meta["max_parallel"]
    queue_policy = (
        "Scoring for this run is serialized. Expect FIFO behavior and do not stop just because the queue is busy."
        if evaluation_mode == "serialized"
        else f"Scoring for this run allows up to {max_parallel} concurrent queued evaluations, but you still must not batch blind future mutations."
    )
    return f"""You own exactly one benchmark run directory.

Your job is to minimize the energy as aggressively as possible while preserving a clean, reproducible mutation history.

Core objective:
- {manifest["objective_text"]}

Hard constraints:
- The untouched baseline must be archived first as iteration 0.
- Then complete exactly iterations 1 through {run_meta["target_iterations"]}.
- One outer iteration = one code mutation + one queued tracked evaluation.
- Use only `evaluate.py` for scored evaluations.
- {queue_policy}
- Inspect the previous scored result before choosing the next mutation.
- If the last result is `discard` or `crash`, restore the best kept iteration before continuing.
- Do not queue batches of future mutations.
- Do not use offline probes, parameter sweeps, or hidden menu-search code paths.

Creativity guidance:
- Be really creative to come up with ways of minimizing the energy.
- Once simple warm-start or tolerance churn stops helping, shift toward better parameterizations, better initial states, staged optimizers, symmetry tying, continuation schedules, or other structural improvements.
- Prefer mathematically coherent changes that can plausibly improve the 20-second fixed-budget score, not cosmetic refactors.
- Keep the method family bounded to the existing benchmark file and nearby support files.
{extra_block}
Before exiting, leave `{manifest["source_file"]}` in the best valid state archived so far and make sure `status.py` reports progress accurately.
"""


def _run_metadata(
    benchmark_root: Path,
    run_dir: Path,
    manifest: dict,
    iterations: int,
    instructions: str,
    evaluation_mode: str,
    max_parallel: int,
) -> dict:
    return {
        "created_at": iso_now(),
        "target_iterations": iterations,
        "additional_instructions": instructions,
        "evaluation_mode": evaluation_mode,
        "max_parallel": max_parallel,
        "benchmark_root": str(benchmark_root),
        "run_dir": str(run_dir),
        "benchmark_value": manifest["benchmark_value"],
        "lane": manifest["lane"],
    }


def resolve_benchmark_dir(repo_root: Path, target: str | None) -> tuple[Path, ExampleSpec]:
    source_map = examples_by_source(repo_root)
    root_map = examples_by_benchmark_root(repo_root)

    if target is None:
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if find_manifest_path(candidate) is not None:
                benchmark_root = _benchmark_root_from_candidate(candidate)
                example = root_map.get(benchmark_root.resolve())
                if example is not None:
                    return benchmark_root, example
        raise FileNotFoundError("no gsopt benchmark found from current directory")

    path = Path(target)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.exists():
        if path.is_file():
            current = path.parent
        else:
            current = path
        for candidate in (current, *current.parents):
            if find_manifest_path(candidate) is not None:
                benchmark_root = _benchmark_root_from_candidate(candidate)
                example = root_map.get(benchmark_root.resolve())
                if example is not None:
                    return benchmark_root, example
        resolved = path.resolve()
        if path.is_file():
            example = source_map.get(resolved)
            if example is not None:
                return _benchmark_root(repo_root, example), example
        for candidate in (current, *current.parents):
            example = root_map.get(candidate.resolve())
            if example is not None:
                return candidate.resolve(), example
    raise FileNotFoundError(f"could not resolve gsopt benchmark from {target}")


def init_run(
    iterations: int,
    target: str | None,
    additional_instructions: str,
    evaluation_mode: str,
    max_parallel: int,
) -> dict:
    repo_root = find_repo_root()
    benchmark_dir, example = resolve_benchmark_dir(repo_root, target)
    manifest = example.manifest_payload()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = benchmark_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_path = benchmark_dir / manifest["source_file"]
    _copy_source_file(source_path, run_dir / manifest["source_file"])
    for name in ("logs", "figs", "snapshots"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    write_json(_manifest_path(run_dir), manifest)
    legacy_manifest = run_dir / ".energyopt.json"
    if legacy_manifest.exists():
        legacy_manifest.unlink()
    for script_name, entrypoint in (
        ("evaluate.py", "evaluate_main"),
        ("restore_best.py", "restore_main"),
        ("plot.py", "plot_main"),
        ("status.py", "status_main"),
    ):
        _write_text(run_dir / script_name, WRAPPER_TEMPLATE.format(entrypoint=entrypoint))
    run_meta = _run_metadata(
        benchmark_dir,
        run_dir,
        manifest,
        iterations,
        additional_instructions,
        evaluation_mode,
        max_parallel,
    )
    write_json(run_dir / "run.json", run_meta)
    write_json(run_dir / "status.json", {})
    _write_text(run_dir / "plan.md", _plan_markdown(manifest, run_meta))
    _write_text(run_dir / "agent_prompt.md", _agent_prompt(manifest, run_meta))
    return {
        "run_dir": str(run_dir),
        "benchmark_dir": str(benchmark_dir),
        "manifest": manifest,
        "run_meta": run_meta,
    }


def sync_benchmark_entrypoints() -> dict:
    repo_root = find_repo_root()
    synced: list[dict[str, str]] = []
    for example in load_examples(repo_root):
        if example.lane.lane not in {"vqe", "tn", "afqmc"}:
            continue
        benchmark_root = _benchmark_root(repo_root, example)
        manifest = example.manifest_payload()
        write_json(_manifest_path(benchmark_root), manifest)
        legacy_manifest = benchmark_root / ".energyopt.json"
        if legacy_manifest.exists():
            legacy_manifest.unlink()
        _write_text(benchmark_root / "evaluate.py", WRAPPER_TEMPLATE.format(entrypoint="evaluate_main"))
        _write_text(benchmark_root / "optuna_baseline.py", WRAPPER_TEMPLATE.format(entrypoint="benchmark_optuna_main"))
        synced.append(
            {
                "lane": example.lane.lane,
                "benchmark": example.benchmark_value,
                "benchmark_root": str(benchmark_root),
            }
        )
    return {"synced": synced}
