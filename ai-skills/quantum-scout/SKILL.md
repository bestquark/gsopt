---
name: quantum-scout
description: Find high-leverage online ideas for lowering fixed-budget quantum benchmark energies. Use when the user wants targeted web or paper research on ansatz design, optimizer schedules, warm starts, tensor-network policies, AFQMC trial states, or similar ways to reach lower energies under compute constraints.
---

# Quantum Scout

Use this skill when the bottleneck is not repo mechanics but idea quality. The job is to find mathematically serious, implementation-relevant ideas that could lower the final energy within the benchmark's fixed wall-time budget.

Adopt the mindset: you are a really smart quantum physicist who knows how to get the lowest-energy states possible under tight compute constraints. Be aggressive, but stay evidence-driven.

## Inputs

Infer these from the repo and user request when not stated explicitly:

- lane and benchmark: VQE, TN, AFQMC, DMRG, or related
- current editable file and current best result
- wall-time budget per evaluation
- whether the search is structural, hyperparameter-only, or both

## Workflow

1. Read the local benchmark file and recent results first.
2. Diagnose the current bottleneck in one line:
   - ansatz too rigid
   - optimizer too slow for the wall budget
   - poor initialization
   - bad continuation schedule
   - too much parameter freedom
   - too little expressive power
3. Do targeted web research only where it can change the search space materially.
4. Prefer primary sources:
   - papers
   - official library docs
   - benchmark repos
5. Translate what you find into concrete ideas that fit the current repo and wall-time budget.
6. Rank ideas by expected payoff per unit implementation complexity.

## Search Targets

Focus the search on ideas like:

- VQE:
  - spin-adapted or symmetry-tied parameterizations
  - pair-doubles / compact UCC variants
  - warm starts from chemically motivated amplitudes
  - staged optimizers and continuation schedules
  - ansatz compression that improves 20-second convergence
- TN / DMRG:
  - stronger initial states
  - better bond-dimension schedules
  - truncation and cutoff heuristics
  - sweep / timestep schedules
  - symmetry-aware structure that improves early convergence
- AFQMC:
  - stronger trial states
  - orbital basis choices
  - timestep / walker / stabilization tradeoffs
  - low-cost changes that reduce bias or variance under fixed runtime

## Search Discipline

- Do not turn this into a broad literature survey.
- Use a small number of focused searches.
- Prioritize ideas that plausibly help within the repo's exact solver stack.
- Reject ideas that require a completely different backend, large offline preprocessing, or a much larger runtime budget.
- When a paper result seems promising but only partially matches the repo, say what is direct evidence and what is your inference.

## Output

Return a compact memo with:

1. current bottleneck
2. 3 to 5 ranked ideas
3. why each idea could help this benchmark under the fixed budget
4. the smallest plausible code mutation for the top 1 or 2 ideas
5. links to the sources used

When used alongside `gsopt`, end with mutation-ready guidance, not just recommendations.
