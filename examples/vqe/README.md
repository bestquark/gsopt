# VQE

Active benchmark directories:

- `bh/`
- `lih/`
- `beh2/`
- `h2o/`
- `n2/`

Each benchmark directory contains:

- `simple_vqe.py`: editable method file
- `evaluate.py`: fixed scorer entrypoint
- `optuna_baseline.py`: separate internal baseline wrapper
- `.gsopt.json`: benchmark metadata for the GSOpt runtime

Smoke test:

```bash
uv run python examples/vqe/bh/simple_vqe.py --wall-seconds 5
```

GSOpt workflow:

```bash
cd examples/vqe/bh
uv run gsopt 100 . "Improve the 20-second final energy."
```

Benchmark-local Optuna baseline:

```bash
uv run python examples/vqe/bh/optuna_baseline.py --wall-seconds 20 --trials 100
```

Figure:

```bash
uv run python figs/vqe/make_energy_figure.py
```
