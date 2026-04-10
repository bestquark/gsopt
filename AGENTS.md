# Repo Guide for External Autoresearch Runs

This repo is organized around `examples/`, `figs/`, `benchkit/`, `skills/`, and `ai-skills/`.

If you want the smallest possible agent-editable targets, prefer the benchmark-local `initial_script.py` files:

- `examples/vqe/bh/initial_script.py`
- `examples/vqe/lih/initial_script.py`
- `examples/vqe/beh2/initial_script.py`
- `examples/vqe/h2o/initial_script.py`
- `examples/vqe/n2/initial_script.py`
- `examples/dmrg/heisenberg_xxx_384/simple_dmrg.py`
- `examples/dmrg/xxz_gapless_256/simple_dmrg.py`
- `examples/dmrg/tfim_longitudinal_256/simple_dmrg.py`
- `examples/dmrg/spin1_heisenberg_64/simple_dmrg.py`
- `examples/dmrg/spin1_single_ion_critical_64/simple_dmrg.py`
- `examples/gibbs/simple_gibbs_mcmc.py`

For VQE, the active external-agent lane is intentionally the five molecule-specific `initial_script.py` files, `track_iteration.py`, `queued_track_iteration.py`, and the root plot script `figs/vqe/make_energy_figure.py`.
For DMRG, the active external-agent lane is intentionally the five model-specific `simple_dmrg.py` files, `track_iteration.py`, `queued_track_iteration.py`, and the root plot script `figs/dmrg/make_energy_figure.py`.

## Setup

1. Run `./bootstrap_cudaq.sh` once at the repo root.
2. Create a timestamped run with `uv run gsopt <num_iterations> [benchmark_dir_or_file] {instructions}`.
3. Run scored evaluations only from inside the generated run directory.
4. Render figures only after runs finish.

Default mutation workflow:

- `uv run gsopt 100 examples/vqe/bh`
- `uv run gsopt run-eval -- uv run python evaluate.py --description "<one-line mutation summary>"`
- `uv run python status.py`
- `uv run python restore_best.py`
- `uv run python plot.py`

## Canonical Lane Commands

- VQE smoke: `uv run python examples/vqe/bh/initial_script.py --wall-seconds 5`
- VQE full evaluation command:
  - `uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/bh/initial_script.py --molecule BH --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/lih/initial_script.py --molecule LiH --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/beh2/initial_script.py --molecule BeH2 --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/h2o/initial_script.py --molecule H2O --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/n2/initial_script.py --molecule N2 --wall-seconds 20 --max-parallel 1`
- DMRG smoke: `uv run python examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --wall-seconds 5`
- DMRG full evaluation command:
  - `uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --model heisenberg_xxx_384 --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/xxz_gapless_256/simple_dmrg.py --model xxz_gapless_256 --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/tfim_longitudinal_256/simple_dmrg.py --model tfim_longitudinal_256 --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/spin1_heisenberg_64/simple_dmrg.py --model spin1_heisenberg_64 --wall-seconds 20 --max-parallel 1`
  - `uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/spin1_single_ion_critical_64/simple_dmrg.py --model spin1_single_ion_critical_64 --wall-seconds 20 --max-parallel 1`
- Gibbs: `uv run python examples/gibbs/search.py examples/gibbs/configs/tfim6.json`

Figures:

- `uv run python figs/vqe/make_energy_figure.py`
- Parallel VQE session prompts: `examples/vqe/instructions/`
- `uv run python figs/dmrg/make_energy_figure.py`
- Parallel DMRG session prompts: `examples/dmrg/instructions/`

## Ground Rules

- Keep claims in the separate paper repo aligned with artifacts that actually exist in `examples/<lane>/<benchmark>/run_<timestamp>/`, `examples/<lane>/<benchmark>/optuna_run_<timestamp>/`, and `figs/<lane>/`.
- Prefer the generated `examples/<lane>/<benchmark>/run_<timestamp>/` directories as the mutation surface for long agent campaigns.
- Use the shared `ai-skills/gsopt/` skill as the canonical workflow wrapper, and `skills/` as the public installer surface for `npx skills add`.
- Prefer editing the lane-specific method file rather than paper text first.
- The active VQE benchmark is a fixed-budget CUDA-Q run on five molecules ordered from easier to harder: `BH`, `LiH`, `BeH2`, `H2O`, and `N2`.
- For VQE external-agent runs, the agent should directly mutate its own molecule-specific `examples/vqe/<molecule>/initial_script.py`, not select from a predeclared ansatz/optimizer menu.
- Each VQE evaluation gets the same 20-second wall-time budget, and the score of interest is the final energy or `\Delta E` at the end of that run.
- Use `examples/vqe/queued_track_iteration.py` for same-machine multi-agent runs; VQE scoring is intentionally serialized through a FIFO on-disk queue so every run sees the same machine state.
- When using `gsopt`, scorer overlap is a run-level policy: `serialized` means `max_parallel=1`, while `parallel` allows a bounded `max_parallel` chosen at run creation.
- The current VQE evaluator caps BLAS/OpenMP thread counts at 10 for each scored run, matching the 10 performance cores of the local Apple M4 Pro machine.
- Use `examples/vqe/track_iteration.py` to archive the exact method file and diff for each accepted outer iteration.
- The intended outer-loop budget is 100 code-mutation iterations per molecule.
- The active DMRG benchmark is a fixed-budget five-model external-agent lane on harder lattice Hamiltonians defined with QuTiP operators and optimized with Quimb DMRG sweeps.
- For DMRG external-agent runs, the agent should directly mutate its own model-specific `examples/dmrg/<model>/simple_dmrg.py`, not select from a config menu.
- Each DMRG evaluation gets the same 20-second wall-time budget, and the score of interest is excess energy relative to a frozen offline high-accuracy DMRG reference.
- Use `examples/dmrg/queued_track_iteration.py` for same-machine multi-agent runs; DMRG scoring is intentionally serialized through a FIFO on-disk queue so every run sees the same machine state.
- The current DMRG evaluator caps BLAS/OpenMP thread counts at 10 for each scored run, matching the 10 performance cores of the local Apple M4 Pro machine.
- Use `examples/dmrg/track_iteration.py` to archive the exact method file and diff for each DMRG outer iteration.
- Archive the untouched DMRG baseline as iteration `0`, then run 100 code-mutation iterations `1` through `100`.
- The Gibbs lane currently optimizes exact-reference quantum state approximation. `simple_gibbs_mcmc.py` is a separate classical Ising MCMC benchmark; do not conflate the two.
