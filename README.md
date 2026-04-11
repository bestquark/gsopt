# gsopt

Shared benchmark repo for fixed-budget energy minimization experiments, internal baselines, and agent-driven mutation runs.

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
- `skills/`: canonical skill bundle and generic runtime
- `benchkit/`: compatibility shims plus internal baseline helpers

`gsopt` is the public skill and CLI name. The generic watchdog, scaffolding,
logging, and mutation-loop runtime now live under `skills/gsopt/scripts/`.
`benchkit/` remains only as a thin compatibility layer for older repo imports
and shared internal baseline helper modules.

The paper sources live separately in `bestquark/quantum_autoresearch`.

## License

MIT. See `LICENSE`.

## Setup

```bash
./bootstrap_cudaq.sh
```

This installs the Python 3.12 environment plus the CUDA-Q CPU backend used by the VQE lane on Apple silicon.

## Quick Start

## Using `/gsopt` In Codex Or Claude Code

The simplest way to use the skill is to open Codex or Claude Code in the
benchmark directory itself, then invoke `/gsopt` with the iteration budget and a
short goal.

Generic toy example for a small H2 VQE folder:

```text
h2_vqe/
├── initial_program.py
└── evaluator.py
```

From inside that folder, run:

```text
/gsopt 50 . --source initial_program.py --evaluator evaluator.py Minimize the H2 VQE energy in 50 scored mutations. Be creative about ansatz structure, initialization, and staged optimizers, but keep the evaluation budget fixed.
```

That tells the agent to:

- scaffold a `run_<timestamp>/` directory
- archive the untouched baseline as iteration `0`
- perform `50` scored mutation iterations
- log every iteration, keep the best state, and use the bundled watchdog flow

Repo-native example with an existing benchmark:

```bash
cd examples/vqe/bh
```

Then in Codex or Claude Code:

```text
/gsopt 50 . Lower the BH VQE final energy as much as possible under the fixed wall-time budget. Prefer structural improvements over seed churn.
```

After the run is scaffolded, the run directory contains the local wrappers the
agent will use:

```bash
python3 run_eval.py -- uv run python evaluate.py --description "archive baseline parameters and starting ansatz"
python3 watchdog.py
python3 campaign.py --agent codex --search
uv run python status.py
uv run python plot.py
```

Smallest direct targets:

```bash
uv run python examples/vqe/bh/simple_vqe.py --wall-seconds 10
uv run python examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --wall-seconds 10
uv run python examples/gibbs/simple_gibbs_mcmc.py
```

Each active benchmark directory now exposes the same local pattern:

- one editable method file such as `simple_vqe.py`, `initial_script.py`, or `simple_dmrg.py`
- `evaluate.py`: benchmark-local entrypoint into the shared tracked evaluator

Internal baselines remain separate from the GSOpt skill and are
not part of the mutation-loop interface.

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
- local `run_eval.py`, `watchdog.py`, and `campaign.py` wrappers that keep working even when the repo-level `gsopt` console entrypoint is unavailable

Inside a run directory:

```bash
python3 run_eval.py -- uv run python evaluate.py --description "archive untouched baseline source"
uv run python status.py
uv run python restore_best.py
uv run python plot.py
python3 watchdog.py
python3 campaign.py --agent codex --search
python3 campaign.py --agent claude
```

For external non-repo benchmarks, the same workflow works on any directory with
an editable method file plus an evaluator file such as `evaluate.py`,
`evaluator.py`, or `eval.py` that prints JSON containing at least a scalar
`score`. If GSOpt cannot infer either file, rerun with `--source <path>` and/or
`--evaluator <path>`.

Internal baseline runs remain separate repo utilities. When you want that
comparison, use the lane-root baseline scripts directly.

`campaign.py` is the strict persistence layer: it repeatedly relaunches Codex or
Claude, checks the logged iteration count after each launch, and keeps waking
the agent until the requested mutation budget is actually completed or repeated
launches make no progress.

Use `--evaluation-mode serialized` for one-at-a-time scoring, or `--evaluation-mode parallel --max-parallel N` when the benchmark should allow bounded concurrency.

## Benchmarks

Active lanes:

- `examples/vqe/`: CUDA-Q molecular VQE
- `examples/tn/`: tensor-network ground-state search
- `examples/afqmc/`: molecular AFQMC, plus the proposed periodic-target roadmap in `examples/afqmc/PERIODIC_TARGETS.md`

Retained but not recently tested:

- `examples/dmrg/`
- `examples/gibbs/`

The main agent-editable targets are the per-benchmark method files such as:

- `examples/vqe/bh/simple_vqe.py`
- `examples/vqe/lih/simple_vqe.py`
- `examples/vqe/beh2/simple_vqe.py`
- `examples/vqe/h2o/simple_vqe.py`
- `examples/vqe/n2/simple_vqe.py`
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
- Internal baseline archives live under per-benchmark archive directories created by the separate baseline runner.
- Historical snapshot archives are expected outside the tracked repo tree. Point plotting scripts at archived roots with the `AUTORESEARCH_*_SNAPSHOT_ROOT` environment variables when needed.
