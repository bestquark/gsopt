# VQE

Active code:

- `bh/simple_vqe.py`
- `lih/simple_vqe.py`
- `beh2/simple_vqe.py`
- `h2o/simple_vqe.py`
- `n2/simple_vqe.py`
- `track_iteration.py`
- `queued_track_iteration.py`
- `figs/vqe/make_energy_figure.py`

Setup:

```bash
./bootstrap_cudaq.sh
```

Each molecule directory also exposes:

- `evaluate.py`

Simple target:

```bash
uv run python examples/vqe/bh/simple_vqe.py --wall-seconds 10
```

External-agent evaluation commands:

```bash
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/bh/simple_vqe.py --molecule BH --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/lih/simple_vqe.py --molecule LiH --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/beh2/simple_vqe.py --molecule BeH2 --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/h2o/simple_vqe.py --molecule H2O --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/n2/simple_vqe.py --molecule N2 --wall-seconds 20 --max-parallel 1
```

Shared Optuna baseline:

```bash
uv run python examples/vqe/optuna_baseline.py --script examples/vqe/bh/simple_vqe.py --molecule BH --wall-seconds 20 --trials 100
```

That creates `examples/vqe/<molecule>/optuna_run_<timestamp>/`.

Figure:

```bash
uv run python figs/vqe/make_energy_figure.py
```

If benchmark-local `optuna_run_<timestamp>/trial_####/result.json` archives exist, the figure script also emits `figs/vqe/vqe_energy_overview_with_optuna.{pdf,png}`.

Benchmark set:

- `BH`: `CAS(2,3)`
- `LiH`: `CAS(2,4)`
- `BeH2`: `CAS(4,4)`
- `H2O`: `CAS(6,4)`
- `N2`: `CAS(6,6)`
