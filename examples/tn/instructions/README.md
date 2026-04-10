# Parallel TN Sessions

Start Codex from `/Users/lmantilla/Desktop/Internship/autoresearch/examples/tn`.
Keep the session in that `tn/` directory.

Open five Codex windows in the same repo tree and use one file per window:
- `heisenberg_xxx_384.md`
- `xxz_gapless_256.md`
- `spin1_heisenberg_64.md`
- `tfim_2d_4x4.md`
- `heisenberg_2d_4x4.md`

Each model should do exactly 100 outer code-mutation iterations.
Use only the queued evaluator with `--wall-seconds 20 --max-parallel 1`.
The queue is FIFO.
Agents may optionally search online for ideas, papers, or implementation hints,
but the local benchmark, file path, and fixed evaluation command must remain the
same.
Do not run offline energy probes, parameter sweeps, or any other non-queued
experiments to guide the search. Outside the queued scorer, only trivial
syntax/import/runtime sanity checks are allowed.
If the target `initial_script.py` file is missing or clearly corrupted, stop and
report it. Do not reconstruct source from `__pycache__`, bytecode, decompilers,
or other reverse-engineering steps.
The untouched baseline must be archived first as iteration `0`, then the
mutated search should continue through iterations `1` to `100`.
Inspect each scored result before launching the next queued evaluation. Do not
launch batches of future iterations or pre-script multiple queued runs ahead of
time.
Do not turn the file into a hidden recipe bank, grid search, or menu of many
internal methods. One outer iteration should test one explicit method.
Scored evaluations are intentionally serialized so every run sees the same
machine and wall-clock budget, and each scored TN run caps BLAS/OpenMP thread
counts at 10.
