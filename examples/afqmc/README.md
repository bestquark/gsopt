## AFQMC Benchmark

This lane benchmarks fixed-wall-time molecular AFQMC using `PySCF + ipie`.
Each molecule directory exposes `initial_script.py`, `evaluate.py`, and `optuna_baseline.py`. The search
space is intentionally bounded to AFQMC-relevant choices such as:

- single-determinant trial family (`rhf` or `uhf`)
- orbital basis (`mo` or `ortho_ao`)
- Cholesky cutoff
- walker count
- timestep
- steps per block
- stabilization / population-control cadence

The five active molecule lanes are:

- `H2`
- `LiH`
- `BH`
- `BeH2`
- `HF`

Each lane uses a fixed 20-second AFQMC production budget per scored run and is scored against a frozen full-CI reference in the same `sto-3g` basis.

## Optuna baseline

Use the benchmark-local Optuna baseline from inside a molecule directory:

```bash
cd examples/afqmc/h2
uv run python optuna_baseline.py --wall-seconds 20 --trials 100
```

That creates `examples/afqmc/<molecule>/optuna_run_<timestamp>/`.

If benchmark-local `optuna_run_<timestamp>/trial_####/result.json` archives exist, `figs/afqmc/make_energy_figure.py` also emits `figs/afqmc/afqmc_error_overview_with_optuna.{pdf,png}`.

## Codex watchdog

For long AFQMC runs that should continue through iteration `100` even if one Codex session exits early, use:

`uv run python run_codex_watchdog.py --molecule H2`

Replace `H2` with any active molecule. The watchdog repeatedly invokes `codex exec`, reads the current snapshot state, resumes from the next required iteration, and stops only when the benchmark reaches the target iteration or repeated launches make no progress.
