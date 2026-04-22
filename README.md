# gsopt

Shared benchmark repo for fixed-budget ground-state optimization experiments, internal baselines, and agent-driven mutation loops.

## Install the Skill

```bash
npx skills add bestquark/gsopt
```

To update an installed skill later, run `npx skills update gsopt -y` or add `-g` for a global install.

For local testing from inside this repo:

```bash
npx skills add .
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

Recommended: use the installed skill from inside Codex or Claude Code. The
skill creates a timestamped `run_<timestamp>/` directory and all monitoring
tools live inside that run directory.

Claude Code slash-command example:

```bash
cd examples/vqe/bh
claude
```

```text
/gsopt 100 . Bias toward structural ansatz improvements.
```

Codex skill example:

```bash
cd examples/vqe/bh
codex
```

```text
$gsopt Run 100 iterations in the current directory. Bias toward structural ansatz improvements.
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

`uv run gsopt ...` only creates `run_<timestamp>/` and the local GSOpt runtime
files. It does not choose or launch the optimizing model by itself. The agent is
whichever Codex or Claude session you use afterward, or whichever agent you
relaunch with `campaign.py` / `async_campaign.py`.

After scaffolding, work inside the run directory:

```bash
python3 run_eval.py -- uv run python evaluate.py --description "archive untouched baseline"
uv run python status.py
uv run python restore_best.py
uv run python plot.py
python3 watchdog.py
python3 tui.py
python3 campaign.py --agent codex --search
python3 async_campaign.py --agent codex --search
python3 campaign.py --agent codex --model <model-name> --search
python3 campaign.py --agent claude --model <model-name>
```

### TUI

Open the live monitor in another terminal:

```bash
cd examples/vqe/bh/run_<timestamp>
python3 tui.py
```

Or from the repo root:

```bash
uv run gsopt tui examples/vqe/bh/run_<timestamp>
uv run gsopt tui examples/vqe/bh/run_<timestamp> --once
```

The TUI shows the target iteration count, latest score, best score, recent
evaluations, and any active local async or Slurm campaign state.

### Async Campaigns

For long evaluations, use the async driver:

```bash
cd examples/vqe/bh/run_<timestamp>
python3 async_campaign.py --agent codex --search
```

The async flow is:

1. Wake Codex or Claude for exactly one source mutation.
2. The agent writes `logs/campaign/pending_mutation.json` and exits.
3. GSOpt runs the scored evaluation while the agent is asleep.
4. When scoring finishes, GSOpt restores the best state if needed and wakes a fresh agent session for the next mutation.

This mirrors the Feynman-style background-work model: durable logs under
`logs/campaign/`, status in `status.json`, and a separate monitor surface via
`python3 tui.py`.

### Slurm

For cluster runs, submit a self-resubmitting async Slurm campaign from inside
the run directory:

```bash
python3 slurm_campaign.py --agent codex --time 04:00:00 --cpus-per-task 12 --mem 32G
python3 slurm_campaign.py --agent claude --partition gpu --gres gpu:1 --setup-command "module load cuda"
```

Each Slurm job performs one async mutation/evaluation step. If the target
mutation count is not done, the job submits the next `sbatch` job with the same
agent and scheduler settings. Slurm state lives under `logs/campaign/slurm/`
and is shown by `python3 tui.py`.

Quickly inspect the mutation history for any run:

```bash
uv run python show_gsopt_log.py examples/afqmc/h2
uv run python show_gsopt_log.py examples/afqmc/h2/run_<timestamp>
```

Each benchmark directory follows the same local pattern:

- one editable method file such as `simple_vqe.py`, `initial_script.py`, or `simple_dmrg.py`
- `evaluate.py` for scored evaluation
- `optuna_baseline.py` for the separate internal baseline
- `.gsopt.json` describing the benchmark to the GSOpt runtime

For VQE and DMRG, the live GSOpt score is the evaluator's `final_energy`. For AFQMC, the live score is the fixed-tail objective `mean_tail + 5 * std_tail`, computed from the final 50% of sampled AFQMC blocks.
Exact-energy error, excess energy, and chemical-accuracy comparisons are kept for offline figures and tables.

If you use GSOpt on a non-repo benchmark, the directory only needs:

- an editable source file
- an evaluator such as `evaluate.py`, `evaluator.py`, or `eval.py` that prints JSON with a scalar `score`

If GSOpt cannot infer either file, rerun with `--source <path>` and/or `--evaluator <path>`.

## Active Benchmarks

- `examples/vqe/`: five molecule-local CUDA-Q VQE benchmarks
- `examples/tn/`: five tensor-network ground-state benchmarks
- `examples/dmrg/`: five model-local DMRG benchmarks
- `examples/afqmc/`: four molecular PySCF + ipie AFQMC benchmarks
- `examples/gibbs/`: separate exact-reference Gibbs / MCMC experiments

Editable targets generally follow:

- `examples/<lane>/<benchmark>/<method-file>`

In practice, each benchmark directory contains one small mutable method file plus fixed scoring infrastructure. Typical method files are:

- `simple_vqe.py` for `examples/vqe/<molecule>/`
- `initial_script.py` for `examples/tn/<model>/` and `examples/afqmc/<molecule>/`
- `simple_dmrg.py` for `examples/dmrg/<model>/`

## Internal Baselines

Optuna is separate from the GSOpt skill. Use the benchmark-local wrappers directly when you want a baseline comparison:

```bash
uv run python examples/vqe/bh/optuna_baseline.py --wall-seconds 20 --trials 100
uv run python examples/tn/heisenberg_xxx_384/optuna_baseline.py --wall-seconds 20 --trials 100
uv run python examples/dmrg/heisenberg_xxx_384/optuna_baseline.py --wall-seconds 20 --trials 100
uv run python examples/afqmc/h2/optuna_baseline.py --wall-seconds 300 --trials 100
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
