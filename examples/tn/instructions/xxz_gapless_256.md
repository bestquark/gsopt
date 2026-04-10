Work only on the `xxz_gapless_256` TN benchmark.

Before doing anything, make sure your current working directory is
`/Users/lmantilla/Desktop/Internship/autoresearch/examples/tn`. Keep the
session in that `tn/` directory.

Edit only:
- `xxz_gapless_256/initial_script.py`

Fixed evaluation command:
`uv run python queued_track_iteration.py --script /Users/lmantilla/Desktop/Internship/autoresearch/examples/tn/xxz_gapless_256/initial_script.py --model xxz_gapless_256 --wall-seconds 20 --max-parallel 1 --description "<one-line mutation summary>"`

Restore-best command:
`uv run python restore_best_iteration.py --script /Users/lmantilla/Desktop/Internship/autoresearch/examples/tn/xxz_gapless_256/initial_script.py --model xxz_gapless_256`

Goal:
- lower the final `final_energy` after exactly 20 seconds

Rules:
1. First run the untouched file through the fixed queued command. That baseline must archive as iteration 0.
2. Then do exactly 100 mutated outer iterations, producing iterations 1 through 100.
3. One outer iteration = one code mutation + one queued 20-second evaluation.
4. After each evaluation, if the status is `discard` or `crash`, run the restore-best command before the next mutation.
5. Inspect the previous scored result before launching the next scored iteration. Do not launch batches of future iterations or pre-script multiple queued runs ahead of time.
6. Do not run direct `initial_script.py` smoke tests for this loop.
7. Do not run direct energy probes, parameter sweeps, or any other non-queued experiments to guide the search.
8. If the target `initial_script.py` file is missing or clearly corrupted, stop and report it.
9. Do not use any `--wall-seconds` other than `20`.
10. Do not change the model Hamiltonian or system size.
11. Keep the method family bounded to the intended tensor-network search space in the file.
12. Ignore unrelated dirty git status entries. Do not edit unrelated files.
13. Waiting in the FIFO queue is expected.
14. Keep the code simple.
15. You may optionally search online for ideas or implementation hints, but keep the local file, model, and fixed evaluation command unchanged.
