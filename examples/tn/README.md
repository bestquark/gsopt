# TN

Active benchmark directories:

- `heisenberg_xxx_384/`
- `xxz_gapless_256/`
- `spin1_heisenberg_64/`
- `tfim_2d_4x4/`
- `heisenberg_2d_4x4/`

Each benchmark directory contains:

- `initial_script.py`: editable method file
- `evaluate.py`: fixed scorer entrypoint
- `optuna_baseline.py`: separate internal baseline wrapper
- `.gsopt.json`: benchmark metadata for the GSOpt runtime

Lane-level shared files kept here:

- `model_registry.py`
- `reference_energies.py`
- `reference_energies.json`
- `compute_reference_energies.py`
- `benchmark_evaluate.py`
- `simple_tn.py`

Smoke test:

```bash
uv run python examples/tn/heisenberg_xxx_384/initial_script.py --wall-seconds 5
```

GSOpt workflow:

```bash
cd examples/tn/heisenberg_xxx_384
uv run gsopt 100 . "Lower the 20-second final energy."
```

Benchmark-local Optuna baseline:

```bash
uv run python examples/tn/heisenberg_xxx_384/optuna_baseline.py --wall-seconds 20 --trials 100
```

Figure:

```bash
uv run python figs/tn/make_energy_figure.py
```
