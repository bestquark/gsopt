---
name: gsopt
description: Run a `/loop`-style fixed-budget code-mutation campaign for ground-state energy or related score minimization. Use when Codex should scaffold a run directory, mutate an editable benchmark file for exactly `n` iterations, score each change through a fixed evaluator such as `evaluate.py` or `evaluator.py`, preserve snapshots/logs, and respect the evaluator's constraints.
---

# GSOpt

Use this skill when the goal is not a one-off code edit but a full mutation loop with a fixed evaluation budget, sequential scored iterations, explicit scorer policy, and complete run logs.

Invocation note:
- In Codex CLI/IDE, skills are invoked with `$gsopt` or through `/skills`, not as a custom `/gsopt` slash command.
- Claude Code may expose the installed skill as `/gsopt`.

Use the generic runtime bundled under `scripts/` for any benchmark directory
that exposes:

- one editable method file, usually `initial_script.py`, `initial_program.py`, or `simple_dmrg.py`
- one evaluator file, usually `evaluate.py`, `evaluator.py`, or `eval.py`, that prints JSON containing `score` or the manifest objective metric
- an optional `.gsopt.json` manifest when lane-specific plot or metadata settings exist

Treat the `examples/` tree in this repo as worked benchmark demonstrations of
the same pattern.

## First step

Create or resume a timestamped run directory before mutating anything:

```bash
uv run gsopt <num_iterations> [benchmark_dir_or_file] {additional instructions}
```

If the repo package is not installed and you are using the skill directly,
invoke the bundled runtime script instead:

```bash
python3 <path-to-installed-gsopt-skill>/scripts/gsopt_cli.py <num_iterations> [benchmark_dir_or_file] {additional instructions}
```

Examples:

```bash
uv run gsopt 100 examples/vqe/bh "Bias toward structural ansatz improvements, not seed churn."
uv run gsopt init-run 100 examples/tn/tfim_2d_4x4 --evaluation-mode parallel --max-parallel 2 "Allow two concurrent evaluations across independent runs."
uv run gsopt 100 . "Be aggressive about better initialization and staged schedules."
uv run gsopt 50 . --source initial_program.py --evaluator evaluator.py "Lower the ground-state energy without changing the evaluator contract."
```

This creates `run_<timestamp>/` under the benchmark directory with:
- a copy of the baseline editable files
- `plan.md`
- `agent_prompt.md`
- `run.json`
- `status.json`
- local `snapshots/`, `logs/`, and `figs/`

After scaffolding, work inside the new run directory.

## Loop contract

1. Read `README.md`, `plan.md`, and `agent_prompt.md`.
2. Archive the untouched baseline first as iteration `0`.
3. Then do exactly the requested number of mutated outer iterations.
4. One outer iteration = inspect the previous result, make one explicit code mutation, and run one scored evaluation.
5. Read the returned JSON before choosing the next mutation.
6. If the last scored result is `discard` or `crash`, restore the best kept iteration before continuing.
7. Keep the live file in the best valid state before you stop.

This is the intended `/loop`-style behavior: one trajectory, one scored step at
a time, with full archival history.

## Evaluation contract

- If no evaluator file is present, stop and tell the user there is no evaluator available for scoring yet. Propose creating one or rerunning GSOpt with `--evaluator <path>`.
- If the editable source file cannot be inferred, stop and ask the user to point to it with `--source <path>`.
- Treat the evaluator as a fixed scoring contract, not as a mutation target.
- Inside a scaffolded run directory, also treat `_user_evaluate.py`, `run_eval.py`, `restore_best.py`, `status.py`, `plot.py`, `watchdog.py`, and `campaign.py` as fixed infrastructure unless the user explicitly asks to change GSOpt itself.
- Mutate the benchmark method file named by the manifest or benchmark path. Only touch other nearby method-support files when the benchmark truly requires a coupled change and the scoring semantics stay identical.
- Optimize the evaluator's returned objective, not an assumed proxy. Depending on the benchmark, the objective may be final energy, `delta E`, absolute energy error, or another lower-is-better score.
- Every scored iteration must include a short technical mutation summary that explicitly names the code change(s) made in that iteration.
- Run scored evaluations only through the wrapped launcher so snapshots, watchdogs, and logs stay consistent.

## Scorer policy

The run metadata records whether scoring is:

- `serialized`: one scored evaluation at a time
- `parallel`: allow up to `max_parallel` concurrent evaluations when the benchmark/runtime supports it

Even in `parallel` mode, do not blindly batch future mutations from one trajectory. Read the previous scored result before choosing the next change.

## Required commands

Run scored evaluations only through the wrapped launcher:

```bash
python3 run_eval.py -- uv run python evaluate.py --description "<technical mutation summary>"
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
python3 watchdog.py
```

If you need a hard guarantee that the agent keeps coming back until the full
mutation budget is finished, use the bundled campaign driver:

```bash
python3 campaign.py --agent codex --search
python3 campaign.py --agent claude
```

This repeatedly relaunches the agent, checks the logged iteration count, and
resumes from the next required mutation until `status.py` reports that the run
hit its target iteration count.

## Search behavior

- Be really creative about lowering the scored objective, usually by improving ground-state energy or its error under a fixed wall-clock budget.
- Prefer coherent ideas that plausibly help the fixed-budget score: better parameterizations, better initial states, staged optimizers, continuation schedules, symmetry tying, dimensionality reduction, or cleaner ansatz structure.
- Once tiny tolerance or seed tweaks plateau, stop burning iterations on them alone.
- Do not batch future evaluations, write menu-search code, or run offline probes outside the scorer.
- Keep the benchmark family intact. Improve the method inside the benchmark; do not replace it with a different solver stack or rewrite the evaluator to make the score easier.

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
- `python3 run_eval.py -- ...` uses the bundled heartbeat and watchdog scripts to kill stalled scored jobs.
- If a run stalls at the campaign level, use `python3 watchdog.py` inside the run directory to detect no-progress periods.
- `python3 campaign.py --agent codex|claude` is the active relaunch loop that wakes the agent again after each launch until the mutation budget is exhausted or repeated launches make no progress.
- `logs/evaluations.jsonl` and each snapshot `metadata.json` store the technical mutation summary passed through `--description`.

## Scope

This skill is the canonical interface across the repo’s benchmark files under `examples/`:
- `examples/vqe/...`
- `examples/tn/...`
- `examples/dmrg/...`
- `examples/afqmc/...`

Prefer opening the benchmark directory itself and using the local editable
method file plus `evaluate.py` interface when it exists. After scaffolding,
continue inside the generated `run_<timestamp>/` directory rather than editing
the shared benchmark file directly.
