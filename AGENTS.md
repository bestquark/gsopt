# Repo Guide for External Autoresearch Runs

This repo is organized around `examples/`, `figs/`, `skills/`, and `benchkit/`.

If you want the smallest agent-editable targets, prefer the benchmark-local method files:

- `examples/vqe/bh/simple_vqe.py`
- `examples/vqe/lih/simple_vqe.py`
- `examples/vqe/beh2/simple_vqe.py`
- `examples/vqe/h2o/simple_vqe.py`
- `examples/vqe/n2/simple_vqe.py`
- `examples/tn/heisenberg_xxx_384/initial_script.py`
- `examples/tn/xxz_gapless_256/initial_script.py`
- `examples/tn/spin1_heisenberg_64/initial_script.py`
- `examples/tn/tfim_2d_4x4/initial_script.py`
- `examples/tn/heisenberg_2d_4x4/initial_script.py`
- `examples/dmrg/heisenberg_xxx_384/simple_dmrg.py`
- `examples/dmrg/xxz_gapless_256/simple_dmrg.py`
- `examples/dmrg/tfim_longitudinal_256/simple_dmrg.py`
- `examples/dmrg/spin1_heisenberg_64/simple_dmrg.py`
- `examples/dmrg/spin1_single_ion_critical_64/simple_dmrg.py`
- `examples/afqmc/h4_square_pbc/initial_script.py`
- `examples/afqmc/h10_chain_pbc/initial_script.py`
- `examples/afqmc/lih_cubic_pbc/initial_script.py`
- `examples/afqmc/diamond_prim/initial_script.py`
- `examples/gibbs/simple_gibbs_mcmc.py`

## Setup

1. Run `./bootstrap_cudaq.sh` once at the repo root.
2. Create a timestamped run with `uv run gsopt <num_iterations> [benchmark_dir_or_file] {instructions}`.
3. Run scored evaluations only from inside the generated run directory.
4. Render figures only after runs finish.

Default mutation workflow:

- `uv run gsopt 100 examples/vqe/bh`
- `python3 run_eval.py -- uv run python evaluate.py --description "<one-line mutation summary>"`
- `python3 campaign.py --agent codex --search`
- `uv run python status.py`
- `uv run python restore_best.py`
- `uv run python plot.py`

## Canonical Lane Commands

- VQE smoke: `uv run python examples/vqe/bh/simple_vqe.py --wall-seconds 5`
- TN smoke: `uv run python examples/tn/heisenberg_xxx_384/initial_script.py --wall-seconds 5`
- DMRG smoke: `uv run python examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --wall-seconds 5`
- AFQMC smoke: `uv run python examples/afqmc/h4_square_pbc/initial_script.py --wall-seconds 5`
- Gibbs: `uv run python examples/gibbs/search.py examples/gibbs/configs/tfim6.json`

Figures:

- `uv run python figs/vqe/make_energy_figure.py`
- `uv run python figs/tn/make_energy_figure.py`
- `uv run python figs/dmrg/make_energy_figure.py`
- `uv run python figs/afqmc/make_energy_figure.py`

## Ground Rules

- Keep claims in the separate paper repo aligned with artifacts that actually exist in `examples/<lane>/<benchmark>/run_<timestamp>/`, `examples/<lane>/<benchmark>/optuna_run_<timestamp>/`, and `figs/<lane>/`.
- Prefer the generated `examples/<lane>/<benchmark>/run_<timestamp>/` directories as the mutation surface for long agent campaigns.
- Use `campaign.py` inside a run directory when you need an outer relaunch loop that keeps waking Codex or Claude until the target iteration count is reached.
- Use `skills/gsopt/` as the canonical workflow wrapper and runtime exposed through `npx skills add`.
- Treat `benchkit/` as shared internal baseline helpers and compatibility glue, not the primary mutation-loop surface.
- Prefer editing the lane-specific method file rather than paper text first.
- The active VQE benchmark is the fixed-budget five-molecule CUDA-Q suite: `BH`, `LiH`, `BeH2`, `H2O`, and `N2`.
- The active TN benchmark is the fixed-budget five-model tensor-network suite: `heisenberg_xxx_384`, `xxz_gapless_256`, `spin1_heisenberg_64`, `tfim_2d_4x4`, and `heisenberg_2d_4x4`.
- The active DMRG benchmark is the fixed-budget five-model DMRG suite: `heisenberg_xxx_384`, `xxz_gapless_256`, `tfim_longitudinal_256`, `spin1_heisenberg_64`, and `spin1_single_ion_critical_64`.
- The active AFQMC benchmark is the fixed-budget four-system periodic-electronic suite: `h4_square_pbc`, `h10_chain_pbc`, `lih_cubic_pbc`, and `diamond_prim`.
- For GSOpt runs, the agent should mutate the benchmark-local method file and treat `evaluate.py` as fixed scoring infrastructure.
- Each scored benchmark evaluation uses the same 20-second wall-time budget within a lane unless the benchmark explicitly documents otherwise.
- GSOpt owns iteration tracking, best-state restore, run-local logging, and the watchdog/campaign flow.
- The Gibbs lane currently optimizes exact-reference quantum state approximation. `simple_gibbs_mcmc.py` is a separate classical Ising MCMC benchmark; do not conflate the two.
- Historical snapshot archives are expected outside the tracked repo tree. Point plotting scripts at those archives with `AUTORESEARCH_*_SNAPSHOT_ROOT` when needed.
