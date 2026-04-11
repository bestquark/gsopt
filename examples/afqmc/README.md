## AFQMC Benchmark

This lane benchmarks fixed-wall-time molecular AFQMC using `PySCF + ipie`.
Each molecule directory exposes `initial_script.py` and `evaluate.py`. The search
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

## Next periodic targets

The current AFQMC implementation is still the validated molecular lane used by
the paper. The next nonmolecular extension should move to four compact
periodic-electronic systems:

- `heg_14e_rs1_gamma`: 3D homogeneous electron gas as a clean metallic benchmark
- `h10_chain_pbc`: periodic hydrogen chain as a correlation-sensitive bond-stretching benchmark
- `lih_rocksalt_prim`: LiH primitive cell as a compact ionic insulator
- `diamond_prim`: diamond primitive cell as a compact covalent semiconductor

See [PERIODIC_TARGETS.md](/Users/lmantilla/Desktop/Internship/gsopt/examples/afqmc/PERIODIC_TARGETS.md) for the rationale behind this proposed set.

## Optuna baseline

Use the shared lane-level Optuna baseline:

```bash
uv run python examples/afqmc/optuna_baseline.py --script examples/afqmc/h2/initial_script.py --molecule H2 --wall-seconds 20 --trials 100
```

That creates `examples/afqmc/<molecule>/optuna_run_<timestamp>/`.

If benchmark-local `optuna_run_<timestamp>/trial_####/result.json` archives exist, `figs/afqmc/make_energy_figure.py` also emits `figs/afqmc/afqmc_error_overview_with_optuna.{pdf,png}`.

## Codex watchdog

For long AFQMC runs that should continue through iteration `100` even if one Codex session exits early, use:

`uv run python run_codex_watchdog.py --molecule H2`

Replace `H2` with any active molecule. The watchdog repeatedly invokes `codex exec`, reads the current snapshot state, resumes from the next required iteration, and stops only when the benchmark reaches the target iteration or repeated launches make no progress.
