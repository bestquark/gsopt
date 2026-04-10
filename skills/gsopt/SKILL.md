---
name: gsopt
description: Optimize benchmark energy with a reproducible mutation loop, watchdogs, per-run archives, and targeted online research when better ideas are needed. Use when the user wants iterative code evolution for VQE, tensor networks, DMRG, AFQMC, or similar fixed-budget energy minimization examples.
---

# GSOpt

Use this skill when the goal is not a one-off code edit but a full energy-minimization campaign with a fixed evaluation budget, sequential mutations, explicit scorer policy, and complete run logs.

## First step

Create or resume a timestamped run directory before mutating anything:

```bash
uv run gsopt <num_iterations> [benchmark_dir_or_file] {additional instructions}
```

Examples:

```bash
uv run gsopt 100 examples/vqe/bh "Bias toward structural ansatz improvements, not seed churn."
uv run gsopt init-run 100 examples/tn/tfim_2d_4x4 --evaluation-mode parallel --max-parallel 2 "Allow two concurrent queued evaluations across independent runs."
uv run gsopt 100 . "Be aggressive about better initialization and staged schedules."
```

Each benchmark directory now exposes:
- `initial_script.py`
- `evaluate.py`
- `optuna_baseline.py`

This creates `examples/.../run_<timestamp>/` with:
- a copy of the baseline editable files
- `plan.md`
- `agent_prompt.md`
- `run.json`
- `status.json`
- local `snapshots/`, `logs/`, and `figs/`

After scaffolding, work inside the new run directory.

## Workflow

1. Read `README.md`, `plan.md`, and `agent_prompt.md`.
2. Archive the untouched baseline first as iteration `0`.
3. Then do exactly the requested number of mutated outer iterations.
4. One outer iteration = one explicit code mutation plus one queued scored evaluation.
5. Read the previous scored result before choosing the next mutation.
6. If the last scored result is `discard` or `crash`, restore the best kept iteration before continuing.
7. Keep the live file in the best valid state before you stop.

## Scorer policy

The run metadata records whether scoring is:

- `serialized`: one queued evaluation at a time
- `parallel`: allow up to `max_parallel` queued evaluations

Even in `parallel` mode, do not blindly batch future mutations from one trajectory. Read the previous scored result before choosing the next change.

## Required commands

Run scored evaluations only through the wrapped launcher:

```bash
uv run gsopt run-eval -- uv run python evaluate.py --description "<one-line mutation summary>"
```

Check status:

```bash
uv run python status.py
```

Render local figures:

```bash
uv run python plot.py
```

Restore the best kept iteration:

```bash
uv run python restore_best.py
```

Optional run-level watchdog in a second terminal:

```bash
uv run gsopt watchdog .
```

## Search behavior

- Be really creative to come up with ways of minimizing the energy.
- Prefer coherent ideas that plausibly help the fixed-budget score: better parameterizations, better initial states, staged optimizers, continuation schedules, symmetry tying, dimensionality reduction, or cleaner ansatz structure.
- Once tiny tolerance or seed tweaks plateau, stop burning iterations on them alone.
- Do not batch future evaluations, write menu-search code, or run offline probes outside the queued scorer.
- Keep the benchmark family intact. Improve the method inside the benchmark; do not replace it with a different solver stack.

## Idea research mode

When the search space feels stale or underinformed, do targeted online research
before spending many more scored iterations. Adopt the mindset: you are a really
smart quantum physicist who knows how to get the lowest-energy states possible
under tight compute constraints. Be aggressive, but stay evidence-driven.

Use targeted research only where it can materially change the mutation space:

- diagnose the current bottleneck first: ansatz rigidity, bad initialization,
  optimizer inefficiency, overparameterization, underexpressivity, or weak
  continuation schedules
- prefer primary sources: papers, official docs, and benchmark repos
- translate what you find into concrete mutations that fit the repo's exact
  solver stack and fixed wall-time budget
- reject ideas that require a different backend, heavy offline preprocessing, or
  a much larger runtime budget

Focus the search on ideas like:

- VQE: spin-adapted or symmetry-tied parameterizations, pair-doubles / compact
  UCC variants, chemically motivated warm starts, staged optimizers, and ansatz
  compression that improves 20-second convergence
- TN / DMRG: stronger initial states, better bond-dimension schedules,
  truncation/cutoff heuristics, timestep schedules, and symmetry-aware structure
- AFQMC: stronger trial states, orbital basis choices, timestep / walker /
  stabilization tradeoffs, and low-cost changes that reduce bias or variance
  under fixed runtime

If you research, keep the output compact and mutation-ready:

1. current bottleneck
2. 3 to 5 ranked ideas
3. why each idea could help under the fixed budget
4. the smallest plausible code mutation for the top 1 or 2 ideas
5. links to the sources used

## Logging and recovery

- `evaluate.py` appends each scored evaluation to `logs/evaluations.jsonl`.
- `status.py` reports how many mutated iterations are completed and whether the run hit its target.
- `uv run gsopt run-eval -- ...` uses the bundled heartbeat and watchdog scripts to kill stalled scored jobs.
- If a run stalls at the campaign level, use `uv run gsopt watchdog .` to detect no-progress periods.

## Scope

This skill is the canonical interface across the repo’s benchmark files under `examples/`:
- `examples/vqe/...`
- `examples/tn/...`
- `examples/dmrg/...`
- `examples/afqmc/...`

For VQE, TN, and AFQMC, prefer opening the benchmark directory itself and using the local `initial_script.py` / `evaluate.py` / `optuna_baseline.py` interface. After scaffolding, continue inside the generated `run_<timestamp>/` directory rather than editing the shared benchmark file directly.
