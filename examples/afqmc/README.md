# AFQMC

The AFQMC lane now uses four molecular PySCF + `ipie` benchmarks with live
phaseless AFQMC evaluations and offline `CCSD(T)` references.

Active benchmark directories:

- `h2/`
- `lih/`
- `h2o/`
- `n2/`

Each benchmark directory contains:

- `initial_script.py`: editable AFQMC method file
- `evaluate.py`: fixed scorer entrypoint
- `optuna_baseline.py`: separate internal baseline wrapper
- `.gsopt.json`: benchmark metadata for the GSOpt runtime

Lane-level shared files:

- `model_registry.py`
- `molecular_benchmark.py`
- `reference_energies.py`
- `reference_energies.json`
- `compute_reference_energies.py`
- `update_reference_source.py`
- `benchmark_evaluate.py`

The live score is the fixed-tail AFQMC objective:

```text
score = mean_tail + 5 * std_tail
```

where `mean_tail` and `std_tail` are computed from the final 50% of sampled
AFQMC blocks.

`CCSD(T)` error is an offline comparison metric only.

## Cluster setup

AFQMC evaluations are expected to run under an MPI launcher on the cluster.
Set the launcher prefix once in the shell before running scored evaluations:

```bash
export AUTORESEARCH_AFQMC_MPI_LAUNCH="mpirun -n 12"
```

If your cluster uses `srun`, use the corresponding launch prefix instead.

## References

If `reference_energies.json` is missing or stale:

```bash
uv run python examples/afqmc/compute_reference_energies.py
```

If you later compute an external offline benchmark, such as a higher-cost
AFQMC production reference, merge it into the same file with:

```bash
uv run python examples/afqmc/update_reference_source.py \
  --system n2 \
  --method-key ph_afqmc \
  --method-label "ph-AFQMC" \
  --energy -107.623456789 \
  --stderr 0.00012 \
  --wall-seconds 1842 \
  --primary
```

## Smoke test

```bash
AUTORESEARCH_AFQMC_MPI_LAUNCH="mpirun -n 12" \
uv run python examples/afqmc/h2/evaluate.py --wall-seconds 300
```

## GSOpt workflow

```bash
cd examples/afqmc/h2
codex
```

```text
$gsopt Run 100 iterations in the current directory. Lower the 5-minute AFQMC score mean_tail + 5 * std_tail over the last 50% of blocks without changing the evaluator contract.
```

Claude Code may expose the same skill as:

```text
/gsopt 100 . Lower the 5-minute AFQMC score mean_tail + 5 * std_tail over the last 50% of blocks without changing the evaluator contract.
```

Manual scaffolding fallback:

```bash
cd examples/afqmc/h2
uv run gsopt 100 . "Lower the 5-minute AFQMC score mean_tail + 5 * std_tail over the last 50% of blocks."
```

Benchmark-local Optuna baseline:

```bash
AUTORESEARCH_AFQMC_MPI_LAUNCH="mpirun -n 12" \
uv run python examples/afqmc/h2/optuna_baseline.py --wall-seconds 300 --trials 100
```

Figures:

```bash
uv run python figs/afqmc/make_energy_figure.py
uv run python figs/afqmc/make_violin_energy_figure.py
uv run python figs/afqmc/make_block_trace_figure.py
```
