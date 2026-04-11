# DMRG

Active benchmark directories:

- `heisenberg_xxx_384/`
- `xxz_gapless_256/`
- `tfim_longitudinal_256/`
- `spin1_heisenberg_64/`
- `spin1_single_ion_critical_64/`

Each benchmark directory contains:

- `simple_dmrg.py`: editable method file
- `evaluate.py`: fixed scorer entrypoint
- `optuna_baseline.py`: separate internal baseline wrapper
- `.gsopt.json`: benchmark metadata for the GSOpt runtime

Lane-level shared files kept here:

- `model_registry.py`
- `reference_energies.py`
- `reference_energies.json`
- `compute_reference_energies.py`
- `benchmark_evaluate.py`

Smoke test:

```bash
uv run python examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --wall-seconds 5
```

GSOpt workflow:

```bash
cd examples/dmrg/heisenberg_xxx_384
uv run gsopt 100 . "Lower the 20-second final energy."
```

Benchmark-local Optuna baseline:

```bash
uv run python examples/dmrg/heisenberg_xxx_384/optuna_baseline.py --wall-seconds 20 --trials 100
```

Figure:

```bash
uv run python figs/dmrg/make_energy_figure.py
```
