# Tensor-Network Ground-State Benchmark

This benchmark is the broader classical counterpart to the VQE suite. Each model directory exposes an editable `initial_script.py` plus a local `evaluate.py` wrapper.

Current allowed method families:

- `dmrg1`
- `dmrg2`
- `tebd1d`
- `tebd2d`

Current active models:

- `heisenberg_xxx_384`
- `xxz_gapless_256`
- `spin1_heisenberg_64`
- `tfim_2d_4x4`
- `heisenberg_2d_4x4`

Each lane follows the same outer-loop protocol as the other benchmarks:

- archive the untouched baseline as iteration `0`
- then run `100` mutated outer iterations
- use only the queued scorer
- restore the best snapshot after any discard or crash

Generated directories:

- `snapshots/`: archived iterations when using the lane-level tracker
- `eval_queue/`: shared TN scorer queue
- `instructions/`: one prompt per lane
- `reference_energies.json`: frozen offline references once computed

Current figure script:

- `uv run python figs/tn/make_energy_figure.py`

Shared Optuna baseline example:

```bash
uv run python examples/tn/optuna_baseline.py --script examples/tn/heisenberg_xxx_384/initial_script.py --model heisenberg_xxx_384 --wall-seconds 20 --trials 100
```

That creates `examples/tn/<model>/optuna_run_<timestamp>/`.

If benchmark-local `optuna_run_<timestamp>/trial_####/result.json` archives exist, `make_energy_figure.py` also emits `figs/tn/tn_energy_overview_with_optuna.{pdf,png}`.

Current queue viewer:

- `uv run python benchkit/show_queues.py --follow --lane tn`
