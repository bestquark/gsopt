Work only on the `xxz_gapless_256` DMRG benchmark.

Before doing anything, make sure your current working directory is
`/Users/lmantilla/Desktop/Internship/autoresearch/examples/dmrg`. Keep the
session in that `dmrg/` directory.

Edit only:
- `xxz_gapless_256/simple_dmrg.py`

Fixed evaluation command:
`uv run python queued_track_iteration.py --script /Users/lmantilla/Desktop/Internship/autoresearch/examples/dmrg/xxz_gapless_256/simple_dmrg.py --model xxz_gapless_256 --wall-seconds 20 --max-parallel 1 --description "<one-line mutation summary>"`

Restore-best command:
`uv run python restore_best_iteration.py --script /Users/lmantilla/Desktop/Internship/autoresearch/examples/dmrg/xxz_gapless_256/simple_dmrg.py --model xxz_gapless_256`

Goal:
- lower the final `final_energy` after exactly 20 seconds

Rules:
1. First run the untouched file through the fixed queued command. That baseline must archive as iteration 0.
2. Then do exactly 100 mutated outer iterations, producing iterations 1 through 100.
3. One outer iteration = one code mutation + one queued 20-second evaluation.
4. After each evaluation, if the status is `discard` or `crash`, run the restore-best command before the next mutation.
5. Inspect the previous scored result before launching the next scored iteration. Do not launch batches of future iterations or pre-script multiple queued runs ahead of time.
6. Do not run direct `simple_dmrg.py` smoke tests for this loop.
7. Do not run direct energy probes, parameter sweeps, or any other non-queued experiments to guide the search. Outside the queued scorer, only trivial syntax/import/runtime sanity checks are allowed.
8. If the target `simple_dmrg.py` file is missing or clearly corrupted, stop and report it. Do not reconstruct source from `__pycache__`, bytecode, decompilers, or other reverse-engineering steps.
9. Do not use any `--wall-seconds` other than `20`.
10. Do not change the model Hamiltonian or chain length.
11. Do not turn the file into a hidden recipe bank, grid search, or menu of many internal methods. One outer iteration should test one explicit method.
12. Ignore unrelated dirty git status entries. Do not edit unrelated files.
13. Waiting in the FIFO queue is expected. Do not stop just because the queue is busy.
14. Keep the code simple.
15. You may optionally search online for ideas or implementation hints, but keep the local file, model, and fixed evaluation command unchanged.
