# DMRG

Status: retained in the repo, but not recently tested after the single-tree `gsopt` reorganization.

Active code:

- `simple_dmrg.py`: generic hard-model definitions and DMRG method
- `heisenberg_xxx_384/simple_dmrg.py`
- `xxz_gapless_256/simple_dmrg.py`
- `tfim_longitudinal_256/simple_dmrg.py`
- `spin1_heisenberg_64/simple_dmrg.py`
- `spin1_single_ion_critical_64/simple_dmrg.py`
- `track_iteration.py`
- `queued_track_iteration.py`
- `figs/dmrg/make_energy_figure.py`

Simple target:

```bash
uv run python examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --wall-seconds 10
```

Full external-agent evaluation commands:

```bash
uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/heisenberg_xxx_384/simple_dmrg.py --model heisenberg_xxx_384 --wall-seconds 20 --max-parallel 1
uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/xxz_gapless_256/simple_dmrg.py --model xxz_gapless_256 --wall-seconds 20 --max-parallel 1
uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/tfim_longitudinal_256/simple_dmrg.py --model tfim_longitudinal_256 --wall-seconds 20 --max-parallel 1
uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/spin1_heisenberg_64/simple_dmrg.py --model spin1_heisenberg_64 --wall-seconds 20 --max-parallel 1
uv run python examples/dmrg/queued_track_iteration.py --script examples/dmrg/spin1_single_ion_critical_64/simple_dmrg.py --model spin1_single_ion_critical_64 --wall-seconds 20 --max-parallel 1
```

Benchmark set:

- `heisenberg_xxx_384`
- `xxz_gapless_256`
- `tfim_longitudinal_256`
- `spin1_heisenberg_64`
- `spin1_single_ion_critical_64`
