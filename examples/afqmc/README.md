# AFQMC

The AFQMC lane uses compact periodic-electronic PySCF-PBC benchmarks with
frozen periodic MP2 reference energies.

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
- `benchmark_evaluate.py`

If `reference_energies.json` is missing or stale:

```bash
uv run python examples/afqmc/compute_reference_energies.py
```

Smoke test:

```bash
uv run python examples/afqmc/h8_cube_pbc/initial_script.py --wall-seconds 5
```

GSOpt workflow:

```bash
cd examples/afqmc/h8_cube_pbc
uv run gsopt 100 . "Lower the 20-second absolute energy error."
```

Benchmark-local Optuna baseline:

```bash
uv run python examples/afqmc/h8_cube_pbc/optuna_baseline.py --wall-seconds 20 --trials 100
```

Figure:

```bash
uv run python figs/afqmc/make_energy_figure.py
```
