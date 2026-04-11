# Parallel VQE Sessions

Start Codex from `/Users/lmantilla/Desktop/Internship/autoresearch/examples/vqe`.
Keep the session in that `vqe/` directory.

Open five Codex windows in the same repo tree and use one file per window:
- `bh.md`
- `lih.md`
- `beh2.md`
- `h2o.md`
- `n2.md`

Each molecule should do exactly 100 outer code-mutation iterations.
Use only the queued evaluator with `--wall-seconds 20 --max-parallel 1`.
The queue is FIFO.
Agents may optionally search online for ideas, papers, or implementation hints,
but the local benchmark, file path, and fixed evaluation command must remain the
same.
Do not run offline energy probes, parameter sweeps, direct `cudaq.observe`
studies, or any other non-queued experiments to guide the search. Outside the
queued scorer, only trivial syntax/import/runtime sanity checks are allowed.
If the target `simple_vqe.py` file is missing or clearly corrupted, stop and
report it. Do not reconstruct source from `__pycache__`, bytecode, decompilers,
or other reverse-engineering steps.
The untouched CUDA-Q baseline must be archived first as iteration `0`, then the
mutated search should continue through iterations `1` to `100`.
Chemical accuracy (`1e-3 Ha`) is the main scientific target. If a molecule is
still above chemical accuracy, do not treat tiny warm-start polishing gains as
the real goal. Once local optimizer-radius tweaks or cached-parameter reuse
plateau, prefer larger ansatz, parameterization, or active-subspace changes
that could materially reduce `\Delta E`.
Inspect each scored result before launching the next queued evaluation. Do not
launch batches of future iterations or pre-script multiple queued runs ahead of
time.
Do not write or leave behind detached/background controller scripts that keep
submitting future iterations after the interactive turn ends.
Scored evaluations are intentionally serialized so every run sees the same
machine and wall-clock budget.
