# gsopt

Shared benchmark repo for fixed-budget ground-state optimization experiments, internal baselines, and agent-driven mutation loops.

## Install the Skill

```bash
npx skills add bestquark/gsopt --skill gsopt
```

For local testing from inside this repo:

```bash
npx skills add . --skill gsopt
```

## Layout

- `examples/`: benchmark directories plus small lane-level shared metadata
- `figs/`: plotting scripts
- `skills/`: the public `gsopt` skill and runtime
- `benchkit/`: shared Optuna/baseline helpers and compatibility glue

The mutation-loop runtime lives under `skills/gsopt/scripts/`. Lane-local queue, tracker, restore, and watchdog scripts are no longer the primary workflow surface.

## Setup

```bash
./bootstrap_cudaq.sh
```

## GSOpt Workflow

Recommended: use the installed skill from inside Codex or Claude Code.

Codex example:

```bash
cd examples/vqe/bh
codex
```

```text
$gsopt Run 100 iterations in the current directory. Bias toward structural ansatz improvements.
```

Claude Code example:

```bash
cd examples/vqe/bh
claude
```

```text
/gsopt 100 . Bias toward structural ansatz improvements.
```

You can also target a benchmark from the repo root instead of `cd`-ing first:

```text
$gsopt Run 100 iterations on examples/tn/tfim_2d_4x4. Improve the 20-second final energy.
```

Manual fallback: scaffold the run directory from the shell without invoking an agent yet:

```bash
cd examples/vqe/bh
uv run gsopt 100 . "Bias toward structural ansatz improvements."
```

Or from the repo root:

```bash
uv run gsopt 100 examples/tn/tfim_2d_4x4 "Improve the 20-second final energy."
```

`uv run gsopt ...` only creates `run_<timestamp>/` and the local GSOpt runtime files. It does not choose or launch the optimizing model by itself. The optimizing agent is whichever Codex or Claude session you use afterward, or whichever agent you relaunch with `campaign.py`.

After scaffolding, work inside the run directory:

```bash
python3 run_eval.py -- uv run python evaluate.py --description "archive untouched baseline"
uv run python status.py
uv run python restore_best.py
uv run python plot.py
python3 watchdog.py
python3 campaign.py --agent codex --search
python3 campaign.py --agent codex --model <model-name> --search
python3 campaign.py --agent claude --model <model-name>
```

Quickly inspect the mutation history for any run:

```bash
uv run python show_gsopt_log.py examples/afqmc/h8_cube_pbc
uv run python show_gsopt_log.py examples/afqmc/h8_cube_pbc/run_<timestamp>
```

Each benchmark directory follows the same local pattern:

- one editable method file such as `simple_vqe.py`, `initial_script.py`, or `simple_dmrg.py`
- `evaluate.py` for scored evaluation
- `optuna_baseline.py` for the separate internal baseline
- `.gsopt.json` describing the benchmark to the GSOpt runtime

For VQE, AFQMC, and DMRG, the live GSOpt score is now the evaluator's `final_energy`.
Exact-energy error, excess energy, and chemical-accuracy comparisons are kept for offline figures and tables.

If you use GSOpt on a non-repo benchmark, the directory only needs:

- an editable source file
- an evaluator such as `evaluate.py`, `evaluator.py`, or `eval.py` that prints JSON with a scalar `score`

If GSOpt cannot infer either file, rerun with `--source <path>` and/or `--evaluator <path>`.

## Active Benchmarks

- `examples/vqe/`: five molecule-local CUDA-Q VQE benchmarks
- `examples/tn/`: five tensor-network ground-state benchmarks
- `examples/dmrg/`: five model-local DMRG benchmarks
- `examples/afqmc/`: four periodic-electronic PySCF-PBC benchmarks
- `examples/gibbs/`: separate exact-reference Gibbs / MCMC experiments

Representative editable targets:

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
- `examples/afqmc/h8_cube_pbc/initial_script.py`
- `examples/afqmc/h10_chain_pbc/initial_script.py`
- `examples/afqmc/lih_cubic_pbc/initial_script.py`
- `examples/afqmc/diamond_prim/initial_script.py`

## Internal Baselines

Optuna is separate from the GSOpt skill. Use the benchmark-local wrappers directly when you want a baseline comparison:

```bash
uv run python examples/vqe/bh/optuna_baseline.py --wall-seconds 20 --trials 100
uv run python examples/tn/heisenberg_xxx_384/optuna_baseline.py --wall-seconds 20 --trials 100
uv run python examples/dmrg/heisenberg_xxx_384/optuna_baseline.py --wall-seconds 20 --trials 100
uv run python examples/afqmc/h8_cube_pbc/optuna_baseline.py --wall-seconds 20 --trials 100
```

These create per-benchmark `optuna_run_<timestamp>/` archives.

## Figures

Run the plotters from the repo root:

```bash
uv run python figs/vqe/make_energy_figure.py
uv run python figs/tn/make_energy_figure.py
uv run python figs/dmrg/make_energy_figure.py
uv run python figs/afqmc/make_energy_figure.py
```

Historical snapshot archives are expected outside the tracked repo tree. Point plotting scripts at archived roots with `AUTORESEARCH_*_SNAPSHOT_ROOT` when needed.

## Notes

- Each scored evaluation is fixed-budget and intended to be compared at equal wall time.
- Run-local mutation history lives under `examples/<lane>/<benchmark>/run_<timestamp>/snapshots/`.
- Internal baseline archives live under per-benchmark `optuna_run_<timestamp>/`.
- The paper sources live separately in `bestquark/quantum_autoresearch`.
