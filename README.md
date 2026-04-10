# gsopt

Shared benchmark repo for fixed-budget energy minimization experiments, Optuna baselines, and agent-driven mutation runs.

## Install the `gsopt` Skill

`gsopt` now uses the `vercel-labs/skills` layout directly. The repo's
top-level `skills/` directory is the only canonical skill bundle, and `gsopt`
itself includes the targeted online idea-research guidance.

```bash
npx skills add bestquark/gsopt --skill gsopt
```

For explicit agent targeting, use the flags supported by that CLI, for example:

```bash
npx skills add bestquark/gsopt --skill gsopt -a codex
npx skills add bestquark/gsopt --skill gsopt -a claude-code
```

For local testing from inside the repo itself:

```bash
npx skills add . --skill gsopt
```

## Layout

- `examples/`: canonical benchmark tree
- `figs/`: plotting scripts and generated figures
- `benchkit/`: internal runtime behind `uv run gsopt`
- `skills/`: canonical skill bundle for `npx skills add`

`gsopt` is the public skill and CLI name. `benchkit/` is just the internal package that scaffolds runs, wraps evaluators, and tracks progress.

The paper sources live separately in `bestquark/quantum_autoresearch`.

## Setup

```bash
./bootstrap_cudaq.sh
```

This installs the Python 3.12 environment plus the CUDA-Q CPU backend used by the VQE lane on Apple silicon.

## Quick Start

Smallest direct targets:

```bash
uv run python examples/vqe/bh/initial_script.py --wall-seconds 10
uv run python examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --wall-seconds 10
uv run python examples/gibbs/simple_gibbs_mcmc.py
```

Each active VQE, TN, and AFQMC benchmark directory now exposes the same local interface:

- `initial_script.py`: the benchmark file the agent mutates
- `evaluate.py`: benchmark-local entrypoint into the shared tracked evaluator
- `optuna_baseline.py`: benchmark-local entrypoint into the shared Optuna runner

Canonical mutation workflow from inside a benchmark directory:

```bash
cd examples/vqe/bh
uv run gsopt 100 . "Bias toward structural ansatz improvements."
```

You can also target a benchmark directory from the repo root:

```bash
uv run gsopt 100 examples/vqe/bh "Bias toward structural ansatz improvements."
uv run gsopt init-run 100 examples/tn/tfim_2d_4x4 --evaluation-mode parallel --max-parallel 2 "Allow two concurrent queued evaluations."
```

That creates `examples/<lane>/<benchmark>/run_<timestamp>/` with:

- the copied editable baseline file
- `evaluate.py`, `restore_best.py`, `plot.py`, and `status.py`
- local `snapshots/`, `logs/`, and `figs/`
- `plan.md`, `agent_prompt.md`, `run.json`, and `status.json`

Inside a run directory:

```bash
uv run gsopt run-eval -- uv run python evaluate.py --description "archive untouched baseline"
uv run python status.py
uv run python restore_best.py
uv run python plot.py
uv run gsopt watchdog .
```

Use `--evaluation-mode serialized` for one-at-a-time scoring, or `--evaluation-mode parallel --max-parallel N` when the benchmark should allow bounded concurrency.

## Benchmarks

Active lanes:

- `examples/vqe/`: CUDA-Q molecular VQE
- `examples/tn/`: tensor-network ground-state search
- `examples/afqmc/`: molecular AFQMC, plus the proposed periodic-target roadmap in `examples/afqmc/PERIODIC_TARGETS.md`

Retained but not recently tested:

- `examples/dmrg/`
- `examples/gibbs/`

The main agent-editable targets are the per-benchmark `initial_script.py` files such as:

- `examples/vqe/bh/initial_script.py`
- `examples/vqe/lih/initial_script.py`
- `examples/vqe/beh2/initial_script.py`
- `examples/vqe/h2o/initial_script.py`
- `examples/vqe/n2/initial_script.py`
- `examples/tn/heisenberg_xxx_384/initial_script.py`
- `examples/afqmc/h2/initial_script.py`

## Figures

Run the plotters from the repo root:

```bash
uv run python figs/vqe/make_energy_figure.py
uv run python figs/tn/make_energy_figure.py
uv run python figs/afqmc/make_energy_figure.py
uv run python figs/dmrg/make_energy_figure.py
```

Generated figures land in the matching root directories:

- `figs/vqe/`
- `figs/tn/`
- `figs/afqmc/`
- `figs/dmrg/`

## Notes

- Each tracked evaluation is fixed-budget and intended to be compared on equal wall time.
- Run-local mutation history lives under `examples/<lane>/<benchmark>/run_<timestamp>/snapshots/`.
- Optuna benchmark-local archives live under `examples/<lane>/<benchmark>/optuna_run_<timestamp>/`.
- Shared lane-level archives such as `examples/vqe/snapshots/` remain the source for the paper figures and historical comparisons.
