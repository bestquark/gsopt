Work only on the `BeH2` AFQMC benchmark.

Before doing anything, make sure your current working directory is
`/Users/lmantilla/Desktop/Internship/autoresearch/examples/afqmc`. Keep the
session in that `afqmc/` directory.

Edit only:
- `beh2/initial_script.py`

Fixed evaluation command:
`uv run python queued_track_iteration.py --script /Users/lmantilla/Desktop/Internship/autoresearch/examples/afqmc/beh2/initial_script.py --molecule BeH2 --wall-seconds 20 --max-parallel 1 --description "<one-line mutation summary>"`

Restore-best command:
`uv run python restore_best_iteration.py --script /Users/lmantilla/Desktop/Internship/autoresearch/examples/afqmc/beh2/initial_script.py --molecule BeH2`

Goal:
- lower the final `abs_final_error` after exactly 20 seconds

Rules:
1. First run the untouched file through the fixed queued command. That baseline must archive as iteration 0.
2. Then do exactly 100 mutated outer iterations, producing iterations 1 through 100.
3. One outer iteration = one code mutation + one queued 20-second evaluation.
4. After each evaluation, if the status is `discard` or `crash`, run the restore-best command before the next mutation.
5. Inspect the previous scored result before launching the next scored iteration. Do not launch batches of future iterations or pre-script multiple queued runs ahead of time.
6. Do not run direct `initial_script.py` smoke tests for this loop.
7. Do not run direct energy probes, parameter sweeps, or any other non-queued experiments to guide the search. Outside the queued scorer, only trivial syntax/import/runtime sanity checks are allowed.
8. If the target `initial_script.py` file is missing or clearly corrupted, stop and report it.
9. Do not use any `--wall-seconds` other than `20`.
10. Do not change the molecule geometry or basis.
11. Keep the method family bounded to the intended AFQMC search space in the file. Do not import unrelated solver stacks.
12. Ignore unrelated dirty git status entries. Do not edit unrelated files.
13. Waiting in the FIFO queue is expected. Do not stop just because the queue is busy.
14. Keep the code simple.
15. You may optionally search online for ideas or implementation hints, but keep the local file, molecule, and fixed evaluation command unchanged.
16. If this benchmark is resumed from an existing archive, continue from the next required iteration and keep the sequential loop running through iteration 100 without stopping at turn boundaries.
17. If a persistent attached PTY/controller is needed to survive the session wrapper, that is allowed, but it must still choose the next mutation only after reading the previous scored result. No predefined sweeps, seed loops, or scripted batches of future mutations.
