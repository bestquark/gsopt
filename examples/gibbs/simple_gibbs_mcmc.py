"""
Minimal classical Gibbs-by-MCMC benchmark for external autoresearch agents.

This target is deliberately classical and simple: a 1D Ising chain with an
exactly enumerable Gibbs distribution. The agent should optimize the MCMC
settings to minimize the total-variation distance to the exact distribution.
"""

from __future__ import annotations

import json
import time

import numpy as np

L = 8
BETA = 0.8
J = 1.0
H = 0.3

# The agent should mainly optimize these settings.
NUM_CHAINS = 64
BURN_IN_SWEEPS = 50
SAMPLE_SWEEPS = 200
THINNING = 2
SEED = 42


def enumerate_states(L: int) -> np.ndarray:
    states = np.empty((2**L, L), dtype=np.int8)
    for index in range(2**L):
        bits = [(index >> shift) & 1 for shift in range(L - 1, -1, -1)]
        states[index] = np.array([1 if bit else -1 for bit in bits], dtype=np.int8)
    return states


def energies(states: np.ndarray) -> np.ndarray:
    interaction = -J * np.sum(states[:, :-1] * states[:, 1:], axis=1)
    field = -H * np.sum(states, axis=1)
    return interaction + field


def exact_distribution() -> tuple[np.ndarray, np.ndarray]:
    states = enumerate_states(L)
    e = energies(states)
    logw = -BETA * (e - e.min())
    w = np.exp(logw - logw.max())
    return states, w / w.sum()


def state_index(spins: np.ndarray) -> int:
    idx = 0
    for spin in spins:
        idx = (idx << 1) | int(spin > 0)
    return idx


def delta_energy(spins: np.ndarray, site: int) -> float:
    left = spins[site - 1] if site > 0 else 0
    right = spins[site + 1] if site + 1 < L else 0
    return 2.0 * spins[site] * (J * (left + right) + H)


def empirical_distribution() -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(SEED)
    chains = rng.choice(np.array([-1, 1], dtype=np.int8), size=(NUM_CHAINS, L))
    counts = np.zeros(2**L, dtype=np.float64)

    def metropolis_sweep():
        for chain in chains:
            for site in rng.permutation(L):
                dE = delta_energy(chain, site)
                if dE <= 0.0 or rng.random() < np.exp(-BETA * dE):
                    chain[site] *= -1

    t0 = time.perf_counter()
    for _ in range(BURN_IN_SWEEPS):
        metropolis_sweep()
    for sweep in range(SAMPLE_SWEEPS):
        metropolis_sweep()
        if (sweep + 1) % THINNING == 0:
            for chain in chains:
                counts[state_index(chain)] += 1.0
    runtime_s = time.perf_counter() - t0
    probs = counts / counts.sum()
    return probs, runtime_s


def main():
    states, exact = exact_distribution()
    empirical, runtime_s = empirical_distribution()
    exact_mag = float(np.sum(exact * np.mean(states, axis=1)))
    empirical_mag = float(np.sum(empirical * np.mean(states, axis=1)))
    eps = 1e-12
    summary = {
        "task": "simple_gibbs_mcmc_classical_ising",
        "metric": "total_variation_distance",
        "lower_is_better": True,
        "score": float(0.5 * np.sum(np.abs(empirical - exact))),
        "kl_exact_to_empirical": float(np.sum(exact * np.log((exact + eps) / (empirical + eps)))),
        "magnetization_error": abs(empirical_mag - exact_mag),
        "wall_seconds": runtime_s,
        "L": L,
        "beta": BETA,
        "J": J,
        "h": H,
        "num_chains": NUM_CHAINS,
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "sample_sweeps": SAMPLE_SWEEPS,
        "thinning": THINNING,
        "seed": SEED,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
