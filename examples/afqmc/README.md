# AFQMC

The AFQMC lane uses compact periodic-electronic PySCF-PBC benchmarks with
live periodic MP2 total energies and bootstrap `block_averaged_energies`,
compared offline against frozen periodic references. The current in-repo
deterministic reference is periodic CCSD(T), and the reference file is now
structured so a future `ph-AFQMC` benchmark can be stored alongside it as a
second offline reference.

These offline reference builds are not wall-time capped. The stored
`wall_seconds` fields are measured elapsed runtimes for the completed
reference jobs, not optimization-time budgets.

Active benchmark directories:

- `h8_cube_pbc/`
- `h10_chain_pbc/`
- `lih_cubic_pbc/`
- `diamond_prim/`

Each benchmark directory contains:

- `initial_script.py`: editable method file
- `evaluate.py`: fixed scorer entrypoint
- `optuna_baseline.py`: separate internal baseline wrapper
- `.gsopt.json`: benchmark metadata for the GSOpt runtime

Lane-level shared files kept here:

- `model_registry.py`
- `periodic_benchmark.py`
- `reference_energies.py`
- `reference_energies.json`
- `compute_reference_energies.py`
- `update_reference_source.py`
- `benchmark_evaluate.py`

If `reference_energies.json` is missing or stale:

```bash
uv run python examples/afqmc/compute_reference_energies.py
```

If you compute an external offline benchmark later, such as a high-cost
`ph-AFQMC` reference, merge it into the same file with:

```bash
uv run python examples/afqmc/update_reference_source.py \
  --system h8_cube_pbc \
  --method-key ph_afqmc \
  --method-label "ph-AFQMC" \
  --energy -3.123456789 \
  --stderr 0.00012 \
  --wall-seconds 1842 \
  --primary
```

Smoke test:

```bash
uv run python examples/afqmc/h8_cube_pbc/initial_script.py --wall-seconds 5
```

GSOpt workflow:

```bash
cd examples/afqmc/h8_cube_pbc
codex
```

```text
$gsopt Run 100 iterations in the current directory. Lower the 20-second final energy without changing the evaluator contract.
```

Claude Code may expose the same skill as:

```text
/gsopt 100 . Lower the 20-second final energy without changing the evaluator contract.
```

Manual scaffolding fallback:

```bash
cd examples/afqmc/h8_cube_pbc
uv run gsopt 100 . "Lower the 20-second final energy."
```

Benchmark-local Optuna baseline:

```bash
uv run python examples/afqmc/h8_cube_pbc/optuna_baseline.py --wall-seconds 20 --trials 100
```

Figure:

```bash
uv run python figs/afqmc/make_energy_figure.py
```
