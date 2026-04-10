# Gibbs

Status: retained in the repo, but not recently tested after the single-tree `gsopt` reorganization.

Simple target:

```bash
uv run python examples/gibbs/simple_gibbs_mcmc.py
```

Quantum exact-reference lane:

```bash
uv run python examples/gibbs/search.py examples/gibbs/configs/tfim6.json
```

Smoke run:

```bash
uv run python examples/gibbs/search.py examples/gibbs/configs/smoke.json
```

Progress figure:

```bash
uv run python figs/gibbs/plot_progress.py --run-dir examples/gibbs/runs/gibbs_tfim6_smoke
```
