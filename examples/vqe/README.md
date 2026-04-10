# VQE

Active code:

- `bh/initial_script.py`
- `lih/initial_script.py`
- `beh2/initial_script.py`
- `h2o/initial_script.py`
- `n2/initial_script.py`
- `track_iteration.py`
- `queued_track_iteration.py`
- `figs/vqe/make_energy_figure.py`

Setup:

```bash
./bootstrap_cudaq.sh
```

Each molecule directory also exposes:

- `evaluate.py`
- `optuna_baseline.py`

Simple target:

```bash
uv run python examples/vqe/bh/initial_script.py --wall-seconds 10
```

External-agent evaluation commands:

```bash
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/bh/initial_script.py --molecule BH --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/lih/initial_script.py --molecule LiH --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/beh2/initial_script.py --molecule BeH2 --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/h2o/initial_script.py --molecule H2O --wall-seconds 20 --max-parallel 1
uv run python examples/vqe/queued_track_iteration.py --script examples/vqe/n2/initial_script.py --molecule N2 --wall-seconds 20 --max-parallel 1
```

Benchmark-local Optuna baseline:

```bash
cd examples/vqe/bh
uv run python optuna_baseline.py --wall-seconds 20 --trials 100
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
