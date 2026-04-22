from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .common import (
    find_manifest_path,
    infer_evaluator_file,
    find_repo_root,
    find_skill_root,
    infer_source_file,
    iso_now,
    read_json,
    synthesize_manifest,
    write_json,
)

MANIFEST_NAME = ".gsopt.json"

WRAPPER_TEMPLATE = """from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _locate_runtime_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skills" / "gsopt" / "SKILL.md").exists():
            return candidate / "skills" / "gsopt"
        run_meta = candidate / "run.json"
        if run_meta.exists():
            try:
                payload = json.loads(run_meta.read_text())
            except json.JSONDecodeError:
                payload = {{}}
            runtime_root = payload.get("runtime_root")
            if runtime_root:
                return Path(runtime_root).resolve()
    env_runtime = os.environ.get("GSOPT_RUNTIME_ROOT")
    if env_runtime:
        return Path(env_runtime).resolve()
    raise RuntimeError(f"could not locate gsopt runtime root from {{start}}")

RUNTIME_ROOT = _locate_runtime_root(Path(__file__).resolve())
SCRIPTS_ROOT = RUNTIME_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from gsopt_runtime.entrypoints import {entrypoint}


if __name__ == "__main__":
    raise SystemExit({entrypoint}())
"""

BENCHMARK_EVALUATOR_TEMPLATE = """from __future__ import annotations

import os
import sys
from pathlib import Path


def _locate_repo_root(start: Path) -> Path:
    env_benchmark = os.environ.get("GSOPT_BENCHMARK_ROOT")
    candidates = []
    if env_benchmark:
        benchmark_root = Path(env_benchmark).resolve()
        candidates.extend([benchmark_root, *benchmark_root.parents])
    current = start.resolve()
    if current.is_file():
        current = current.parent
    candidates.extend([current, *current.parents])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError(f"could not locate repo root from {{start}}")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["GSOPT_BENCHMARK_ROOT"] = str(Path(__file__).resolve().parent)

from {module_path} import main as benchmark_evaluate_main


if __name__ == "__main__":
    raise SystemExit(benchmark_evaluate_main(default_source="{source_file}"))
"""

CLI_WRAPPER_TEMPLATE = """from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _locate_runtime_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skills" / "gsopt" / "SKILL.md").exists():
            return candidate / "skills" / "gsopt"
        run_meta = candidate / "run.json"
        if run_meta.exists():
            try:
                payload = json.loads(run_meta.read_text())
            except json.JSONDecodeError:
                payload = {{}}
            runtime_root = payload.get("runtime_root")
            if runtime_root:
                return Path(runtime_root).resolve()
    env_runtime = os.environ.get("GSOPT_RUNTIME_ROOT")
    if env_runtime:
        return Path(env_runtime).resolve()
    raise RuntimeError(f"could not locate gsopt runtime root from {{start}}")


RUNTIME_ROOT = _locate_runtime_root(Path(__file__).resolve())
SCRIPTS_ROOT = RUNTIME_ROOT / "scripts"
RUN_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from gsopt_cli import main as gsopt_main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "{subcommand}", "--run-dir", str(RUN_DIR), *sys.argv[1:]]
    raise SystemExit(gsopt_main())
"""

CONTEXT_CLI_WRAPPER_TEMPLATE = """from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _locate_runtime_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skills" / "gsopt" / "SKILL.md").exists():
            return candidate / "skills" / "gsopt"
        run_meta = candidate / "run.json"
        if run_meta.exists():
            try:
                payload = json.loads(run_meta.read_text())
            except json.JSONDecodeError:
                payload = {{}}
            runtime_root = payload.get("runtime_root")
            if runtime_root:
                return Path(runtime_root).resolve()
    env_runtime = os.environ.get("GSOPT_RUNTIME_ROOT")
    if env_runtime:
        return Path(env_runtime).resolve()
    raise RuntimeError(f"could not locate gsopt runtime root from {{start}}")


RUNTIME_ROOT = _locate_runtime_root(Path(__file__).resolve())
SCRIPTS_ROOT = RUNTIME_ROOT / "scripts"
RUN_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from gsopt_cli import main as gsopt_main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "{subcommand}", str(RUN_DIR), *sys.argv[1:]]
    raise SystemExit(gsopt_main())
"""


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _benchmark_root_from_candidate(candidate: Path) -> Path:
    run_meta_path = candidate / "run.json"
    if run_meta_path.exists():
        return Path(json.loads(run_meta_path.read_text())["benchmark_root"]).resolve()
    return candidate.resolve()


def _benchmark_manifest(
    benchmark_root: Path,
    source_file: Path | None = None,
    evaluator_file: Path | None = None,
) -> dict:
    manifest_path = find_manifest_path(benchmark_root)
    if manifest_path is not None:
        return read_json(manifest_path)
    return synthesize_manifest(benchmark_root, source_file=source_file, evaluator_file=evaluator_file)


def _resolve_local_hint(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"missing {label} file: {candidate}")
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FileNotFoundError(f"{label} file must live inside benchmark root {root}: {candidate}") from exc
    return candidate


def _plan_markdown(manifest: dict, run_meta: dict) -> str:
    source_file = manifest["source_file"]
    evaluator_file = manifest.get("evaluator_file", "evaluate.py")
    objective = manifest["objective_text"]
    evaluation_mode = run_meta["evaluation_mode"]
    max_parallel = run_meta["max_parallel"]
    lane = str(manifest.get("lane", ""))
    if lane == "generic":
        scored_eval_cmd = 'python3 run_eval.py -- python3 evaluate.py --description "<technical mutation summary>"'
        restore_cmd = "python3 restore_best.py"
        status_cmd = "python3 status.py"
        plot_cmd = "python3 plot.py"
        generic_policy = (
            "- Respect the benchmark's stated computational/model constraints. Do not replace the target method with "
            "an exact full-instance solver; if you use exact linear-algebra primitives, keep them fixed-size or "
            "chunked so the total inference cost still matches the benchmark limit."
        )
    else:
        scored_eval_cmd = 'python3 run_eval.py -- uv run python evaluate.py --description "<technical mutation summary>"'
        restore_cmd = "uv run python restore_best.py"
        status_cmd = "uv run python status.py"
        plot_cmd = "uv run python plot.py"
        generic_policy = ""
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

Scored evaluations from inside the run directory:

```bash
{scored_eval_cmd}
```

Restore the best kept iteration:

```bash
{restore_cmd}
```

Check progress:

```bash
{status_cmd}
```

Render local figures:

```bash
{plot_cmd}
```

Optional run-level watchdog in a second terminal:

```bash
python3 watchdog.py
```

Live terminal monitor:

```bash
python3 tui.py
```

Automatic relaunch loop that keeps waking Codex or Claude until the full
mutation budget is finished:

```bash
python3 campaign.py --agent codex --search
```

Async relaunch loop where scored evaluations run while the agent is asleep:

```bash
python3 async_campaign.py --agent codex --search
```

Slurm relaunch loop for clusters:

```bash
python3 slurm_campaign.py --agent codex --time 04:00:00
```

## Search policy

- {queue_line}
- Original evaluator: `{evaluator_file}`.
- Treat the copied evaluator and GSOpt wrappers as fixed scoring infrastructure. Mutate `{source_file}`, not the evaluator.
- Archive the untouched baseline as iteration `0`.
- Then run exactly iterations `1` through `{run_meta["target_iterations"]}`.
- One outer iteration = one explicit code mutation plus one scored run.
- Every scored run must include a short technical `--description` that explicitly names the code mutation(s).
- Optimize the evaluator's returned objective metric, which may be final energy, energy error, or another lower-is-better score.
- Read the last scored result before choosing the next mutation.
- If a scored run is `discard` or `crash`, restore the best kept iteration before continuing.
- Be really creative about reducing the scored objective: once tiny tolerance or seed tweaks plateau, prefer structural changes that alter the search geometry or ansatz quality.
{generic_policy}
- Do not pre-script batches of future iterations, seed sweeps, or offline probes outside the scored evaluator.
"""


def _agent_prompt(manifest: dict, run_meta: dict) -> str:
    extra = run_meta["additional_instructions"].strip()
    extra_block = f"\nAdditional user instructions:\n{extra}\n" if extra else ""
    evaluator_file = manifest.get("evaluator_file", "evaluate.py")
    evaluation_mode = run_meta["evaluation_mode"]
    max_parallel = run_meta["max_parallel"]
    lane = str(manifest.get("lane", ""))
    queue_policy = (
        "Scoring for this run is serialized. Run one scored evaluation at a time."
        if evaluation_mode == "serialized"
        else f"Scoring for this run allows up to {max_parallel} concurrent evaluations, but you still must not batch blind future mutations."
    )
    generic_constraint = (
        "- Respect the benchmark's stated computational/model constraints. Do not replace the target method with an exact "
        "full-instance solver; if you use exact linear-algebra primitives, keep them fixed-size or chunked so the total "
        "inference cost still matches the benchmark limit.\n"
        if lane == "generic"
        else ""
    )
    return f"""You own exactly one benchmark run directory.

Your job is to minimize the scored objective as aggressively as possible while preserving a clean, reproducible mutation history.

Core objective:
- {manifest["objective_text"]}

Hard constraints:
- Treat the original evaluator `{evaluator_file}`, the copied evaluator `_user_evaluate.py`, and the GSOpt wrapper scripts as fixed scoring infrastructure unless the user explicitly asks to change the runtime itself.
- The untouched baseline must be archived first as iteration 0.
- Then complete exactly iterations 1 through {run_meta["target_iterations"]}.
- One outer iteration = one code mutation + one scored evaluation.
- Use only `evaluate.py` for scored evaluations.
- Every scored evaluation must use `--description` with a short technical mutation summary that explicitly names the code changes.
- Optimize the evaluator's returned score/objective, not an assumed proxy metric.
- {queue_policy}
- Inspect the previous scored result before choosing the next mutation.
- If the last result is `discard` or `crash`, restore the best kept iteration before continuing.
- Do not batch future mutations.
- Do not use offline probes, parameter sweeps, or hidden menu-search code paths.
{generic_constraint}

Creativity guidance:
- Be really creative to come up with ways of lowering the score, usually by improving ground-state energy or its error under the fixed budget.
- Once simple warm-start or tolerance churn stops helping, shift toward better parameterizations, better initial states, staged optimizers, symmetry tying, continuation schedules, or other structural improvements.
- Prefer mathematically coherent changes that can plausibly improve the fixed-budget score, not cosmetic refactors.
- Keep the method family bounded to `{manifest["source_file"]}` and only the smallest necessary nearby method-support files. Do not change the scoring contract.
{extra_block}
Before exiting, leave `{manifest["source_file"]}` in the best valid state archived so far and make sure `status.py` reports progress accurately.
"""


def _run_metadata(
    benchmark_root: Path,
    run_dir: Path,
    manifest: dict,
    runtime_root: Path,
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
        "runtime_root": str(runtime_root),
        "benchmark_value": manifest["benchmark_value"],
        "lane": manifest["lane"],
    }


def resolve_benchmark_dir(repo_root: Path, target: str | None) -> tuple[Path, dict]:
    return resolve_benchmark_dir_with_overrides(repo_root, target, None, None)


def resolve_benchmark_dir_with_overrides(
    repo_root: Path,
    target: str | None,
    source_hint: str | None,
    evaluator_hint: str | None,
) -> tuple[Path, dict]:
    if target is None:
        current = Path.cwd().resolve()
    else:
        path = Path(target)
        current = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
        if current.is_file():
            benchmark_root = current.parent
            source_file = current if source_hint is None else _resolve_local_hint(benchmark_root, source_hint, "source")
            evaluator_file = (
                infer_evaluator_file(benchmark_root)
                if evaluator_hint is None
                else _resolve_local_hint(benchmark_root, evaluator_hint, "evaluator")
            )
            return benchmark_root, _benchmark_manifest(
                benchmark_root,
                source_file=source_file,
                evaluator_file=evaluator_file,
            )

    for candidate in (current, *current.parents):
        manifest_path = find_manifest_path(candidate)
        if manifest_path is not None:
            benchmark_root = _benchmark_root_from_candidate(candidate)
            return benchmark_root, read_json(manifest_path)
        try:
            evaluator_file = (
                infer_evaluator_file(candidate)
                if evaluator_hint is None
                else _resolve_local_hint(candidate, evaluator_hint, "evaluator")
            )
        except FileNotFoundError:
            continue
        source_file = (
            infer_source_file(candidate, evaluator_file=evaluator_file)
            if source_hint is None
            else _resolve_local_hint(candidate, source_hint, "source")
        )
        return candidate, _benchmark_manifest(candidate, source_file=source_file, evaluator_file=evaluator_file)
    if evaluator_hint is None:
        raise FileNotFoundError(
            f"could not resolve gsopt benchmark from {target or current}: no evaluator file was found. "
            "Expected one of `evaluate.py`, `evaluator.py`, or `eval.py`. Create one that prints JSON with "
            "`score`, or rerun gsopt with `--evaluator <path>`."
        )
    raise FileNotFoundError(f"could not resolve gsopt benchmark from {target or current}")


def _copy_support_files(repo_root: Path, benchmark_root: Path, manifest: dict, run_dir: Path):
    evaluate_source = benchmark_root / str(manifest.get("evaluator_file", "evaluate.py"))
    if evaluate_source.exists():
        _copy_file(evaluate_source, run_dir / "_user_evaluate.py")
    for support in manifest.get("support_files", []):
        support_path = Path(support)
        if not support_path.is_absolute():
            support_path = (repo_root / support).resolve()
        if support_path.exists():
            relative = support_path.name if support_path.parent == benchmark_root else support
            _copy_file(support_path, run_dir / relative)


def init_run(
    iterations: int,
    target: str | None,
    additional_instructions: str,
    evaluation_mode: str,
    max_parallel: int,
    source_hint: str | None = None,
    evaluator_hint: str | None = None,
) -> dict:
    repo_root = find_repo_root()
    skill_root = find_skill_root()
    benchmark_dir, manifest = resolve_benchmark_dir_with_overrides(repo_root, target, source_hint, evaluator_hint)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = benchmark_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    source_path = benchmark_dir / manifest["source_file"]
    _copy_file(source_path, run_dir / manifest["source_file"])
    _copy_support_files(repo_root, benchmark_dir, manifest, run_dir)
    for name in ("logs", "figs", "snapshots"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    write_json(_manifest_path(run_dir), manifest)

    for script_name, entrypoint in (
        ("evaluate.py", "evaluate_main"),
        ("restore_best.py", "restore_main"),
        ("plot.py", "plot_main"),
        ("status.py", "status_main"),
    ):
        _write_text(run_dir / script_name, WRAPPER_TEMPLATE.format(entrypoint=entrypoint))
    _write_text(run_dir / "run_eval.py", CLI_WRAPPER_TEMPLATE.format(subcommand="run-eval"))
    for script_name, subcommand in (
        ("watchdog.py", "watchdog"),
        ("campaign.py", "campaign"),
        ("async_campaign.py", "async-campaign"),
        ("tui.py", "tui"),
        ("slurm_campaign.py", "slurm-campaign"),
    ):
        _write_text(run_dir / script_name, CONTEXT_CLI_WRAPPER_TEMPLATE.format(subcommand=subcommand))

    run_meta = _run_metadata(
        benchmark_dir,
        run_dir,
        manifest,
        skill_root,
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
    from benchkit.registry import load_examples

    repo_root = find_repo_root()
    synced: list[dict[str, str]] = []
    evaluator_modules = {
        "vqe": "examples.vqe.benchmark_evaluate",
        "tn": "examples.tn.benchmark_evaluate",
        "afqmc": "examples.afqmc.benchmark_evaluate",
        "dmrg": "examples.dmrg.benchmark_evaluate",
    }
    for example in load_examples(repo_root):
        benchmark_root = (repo_root / example.source_template).resolve().parent
        manifest_path = benchmark_root / MANIFEST_NAME
        manifest = example.manifest_payload()
        write_json(manifest_path, manifest)
        lane = str(manifest.get("lane"))
        module_path = evaluator_modules.get(lane)
        if module_path is None:
            continue
        _write_text(
            benchmark_root / "evaluate.py",
            BENCHMARK_EVALUATOR_TEMPLATE.format(
                module_path=module_path,
                source_file=str(manifest["source_file"]),
            ),
        )
        synced.append(
            {
                "lane": lane,
                "benchmark": str(manifest["benchmark_value"]),
                "benchmark_root": str(benchmark_root),
                "manifest_path": str(manifest_path),
            }
        )
    return {"synced": synced}
