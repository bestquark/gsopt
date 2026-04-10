from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn
from scipy.linalg import sqrtm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style, finish_axes, inferno_colors

np.seterr(all="ignore")

LANE_DIR = Path(__file__).resolve().parent
AUTORESEARCH_DIR = LANE_DIR.parent
OUTDIR = LANE_DIR / "benchmark_outputs"
FIGDIR = AUTORESEARCH_DIR / "figs" / "gibbs"
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY = np.eye(2, dtype=np.complex128)


@dataclass
class ThermalCase:
    system: str
    L: int
    beta: float


def build_dense_hamiltonian(system: str, L: int) -> np.ndarray:
    if system == "tfim_critical":
        mpo = qtn.MPO_ham_ising(L, j=1.0, bx=1.0, cyclic=False)
    elif system == "heisenberg_xxz":
        mpo = qtn.MPO_ham_heis(L, j=(1.0, 1.0, 1.0), bz=0.0, cyclic=False)
    else:
        raise ValueError(f"unknown system: {system}")
    return np.asarray(mpo.to_dense(), dtype=np.complex128)


def project_density_matrix(rho: np.ndarray) -> np.ndarray:
    herm = (rho + rho.conj().T) / 2.0
    vals, vecs = np.linalg.eigh(herm)
    vals = np.clip(vals.real, 1e-15, None)
    vals = vals / vals.sum()
    return vecs @ np.diag(vals) @ vecs.conj().T


def matrix_entropy(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(project_density_matrix(rho))
    vals = np.clip(vals.real, 1e-15, 1.0)
    return float(-np.sum(vals * np.log(vals)))


def free_energy(rho: np.ndarray, H: np.ndarray, beta: float) -> float:
    rho = project_density_matrix(rho)
    return float(np.real(np.trace(rho @ H)) - matrix_entropy(rho) / beta)


def exact_gibbs(H: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    evals, evecs = np.linalg.eigh(H)
    log_weights = -beta * (evals - evals.min())
    log_weights = np.clip(log_weights - log_weights.max(), -700.0, 0.0)
    weights = np.exp(log_weights)
    Z = weights.sum()
    rho = evecs @ np.diag(weights / Z) @ evecs.conj().T
    F = float(evals.min() - np.log(Z) / beta)
    return project_density_matrix(rho), F


def low_rank_gibbs(H: np.ndarray, beta: float, rank: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    evals = evals[:rank]
    evecs = evecs[:, :rank]
    log_weights = -beta * (evals - evals.min())
    log_weights = np.clip(log_weights - log_weights.max(), -700.0, 0.0)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    return project_density_matrix(evecs @ np.diag(weights) @ evecs.conj().T)


def local_density(theta_x: float, theta_z: float) -> np.ndarray:
    radius = float(np.hypot(theta_x, theta_z))
    if radius < 1e-14:
        return 0.5 * IDENTITY.copy()
    scale = np.tanh(radius) / radius
    bloch = scale * (theta_x * PAULI_X + theta_z * PAULI_Z)
    return 0.5 * (IDENTITY + bloch)


def local_density_z_only(theta_z: float) -> np.ndarray:
    scale = float(np.tanh(theta_z))
    return 0.5 * (IDENTITY + scale * PAULI_Z)


def kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def product_state_from_thetas(thetas: np.ndarray, L: int, ansatz: str = "xz_local") -> np.ndarray:
    if ansatz == "xz_local":
        mats = [local_density(thetas[2 * i], thetas[2 * i + 1]) for i in range(L)]
    elif ansatz == "z_local":
        mats = [local_density_z_only(thetas[i]) for i in range(L)]
    else:
        raise ValueError(f"unknown ansatz: {ansatz}")
    return project_density_matrix(kron_all(mats))


def variational_product_gibbs(
    H: np.ndarray,
    beta: float,
    L: int,
    ansatz: str = "xz_local",
    optimizer: str = "BFGS",
    maxiter: int = 400,
    init_scale: float = 0.0,
    init_seed: int = 0,
) -> np.ndarray:
    rho, _ = variational_product_gibbs_with_history(
        H,
        beta,
        L,
        ansatz=ansatz,
        optimizer=optimizer,
        maxiter=maxiter,
        init_scale=init_scale,
        init_seed=init_seed,
    )
    return rho


def variational_product_gibbs_with_history(
    H: np.ndarray,
    beta: float,
    L: int,
    ansatz: str = "xz_local",
    optimizer: str = "BFGS",
    maxiter: int = 400,
    init_scale: float = 0.0,
    init_seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(init_seed)
    ndim = 2 * L if ansatz == "xz_local" else L
    x0 = rng.normal(scale=init_scale, size=(ndim,)).astype(np.float64)
    history: list[float] = []

    def objective(x):
        rho = product_state_from_thetas(np.asarray(x, dtype=np.float64), L, ansatz=ansatz)
        value = free_energy(rho, H, beta)
        if not np.isfinite(value):
            return 1e12
        return value

    history.append(objective(x0))

    def callback(xk):
        history.append(objective(np.asarray(xk, dtype=np.float64)))

    options = {"maxiter": maxiter}
    if optimizer == "BFGS":
        options["gtol"] = 1e-7
    res = minimize(objective, x0, method=optimizer, options=options, callback=callback)
    x_best = res.x if np.all(np.isfinite(res.x)) else x0
    final_rho = product_state_from_thetas(np.asarray(x_best, dtype=np.float64), L, ansatz=ansatz)
    if not history:
        history.append(objective(x_best))
    return final_rho, history


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = project_density_matrix(rho) - project_density_matrix(sigma)
    diff = (diff + diff.conj().T) / 2.0
    evals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(evals)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    rho = project_density_matrix(rho)
    sigma = project_density_matrix(sigma)
    root = sqrtm(rho)
    middle = root @ sigma @ root
    val = np.trace(sqrtm((middle + middle.conj().T) / 2.0))
    return float(np.clip(np.real(val) ** 2, 0.0, 1.0))


def magnetization_z(rho: np.ndarray, L: int) -> float:
    rho = project_density_matrix(rho)
    total = np.zeros_like(rho)
    for i in range(L):
        ops = [IDENTITY] * L
        ops[i] = PAULI_Z
        total = total + kron_all(ops)
    return float(np.real(np.trace(rho @ total)) / L)


def evaluate_case(case: ThermalCase) -> list[dict]:
    H = build_dense_hamiltonian(case.system, case.L)
    rho_exact, free_exact = exact_gibbs(H, case.beta)
    energy_exact = float(np.real(np.trace(rho_exact @ H)))
    mag_exact = magnetization_z(rho_exact, case.L)

    methods = {
        "low_rank_2": low_rank_gibbs(H, case.beta, rank=2),
        "low_rank_4": low_rank_gibbs(H, case.beta, rank=4),
        "product_local": variational_product_gibbs(H, case.beta, case.L),
    }

    rows = []
    for method, rho in methods.items():
        energy = float(np.real(np.trace(rho @ H)))
        free = free_energy(rho, H, case.beta)
        mag = magnetization_z(rho, case.L)
        rows.append(
            {
                "system": case.system,
                "L": case.L,
                "beta": case.beta,
                "method": method,
                "energy": energy,
                "exact_energy": energy_exact,
                "energy_error": abs(energy - energy_exact),
                "free_energy": free,
                "free_energy_gap": free - free_exact,
                "trace_distance": trace_distance(rho, rho_exact),
                "fidelity": fidelity(rho, rho_exact),
                "magnetization_z": mag,
                "magnetization_z_error": abs(mag - mag_exact),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    methods = sorted({row["method"] for row in rows})
    out = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        out[method] = {
            "mean_energy_error": float(np.mean([row["energy_error"] for row in subset])),
            "mean_free_energy_gap": float(np.mean([row["free_energy_gap"] for row in subset])),
            "mean_trace_distance": float(np.mean([row["trace_distance"] for row in subset])),
            "mean_fidelity": float(np.mean([row["fidelity"] for row in subset])),
        }
    return {
        "status": "complete",
        "experiment": "experiment3_gibbs_state_preparation",
        "question": "Which low-complexity approximations best match exact Gibbs states on small spin chains?",
        "methods": out,
        "row_count": len(rows),
    }


def write_tsv(rows: list[dict]):
    header = [
        "system",
        "L",
        "beta",
        "method",
        "energy",
        "exact_energy",
        "energy_error",
        "free_energy",
        "free_energy_gap",
        "trace_distance",
        "fidelity",
        "magnetization_z",
        "magnetization_z_error",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["system"]),
                    str(row["L"]),
                    f"{row['beta']:.2f}",
                    str(row["method"]),
                    f"{row['energy']:.12f}",
                    f"{row['exact_energy']:.12f}",
                    f"{row['energy_error']:.12e}",
                    f"{row['free_energy']:.12f}",
                    f"{row['free_energy_gap']:.12e}",
                    f"{row['trace_distance']:.12e}",
                    f"{row['fidelity']:.12f}",
                    f"{row['magnetization_z']:.12f}",
                    f"{row['magnetization_z_error']:.12e}",
                ]
            )
        )
    (OUTDIR / "experiment3_gibbs_results.tsv").write_text("\n".join(lines) + "\n")


def make_main_figure(rows: list[dict]):
    apply_style()
    methods = ["low_rank_2", "low_rank_4", "product_local"]
    labels = {
        "low_rank_2": "Low-rank (k=2)",
        "low_rank_4": "Low-rank (k=4)",
        "product_local": "Variational product state",
    }
    colors = dict(zip(methods, inferno_colors(len(methods))))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    betas = sorted({row["beta"] for row in rows})

    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        mean_td = [np.mean([row["trace_distance"] for row in subset if row["beta"] == beta]) for beta in betas]
        mean_gap = [np.mean([row["free_energy_gap"] for row in subset if row["beta"] == beta]) for beta in betas]
        axes[0].plot(betas, mean_td, marker="o", color=colors[method], label=labels[method])
        axes[1].plot(betas, mean_gap, marker="o", color=colors[method], label=labels[method])

    finish_axes(axes[0], xlabel=r"Inverse temperature $\beta$", ylabel="Trace distance")
    finish_axes(axes[1], xlabel=r"Inverse temperature $\beta$", ylabel=r"$F(\rho)-F(\rho_\beta)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIGDIR / "experiment3_gibbs_quality.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def representative_case() -> ThermalCase:
    return ThermalCase("tfim_critical", 6, 2.0)


def make_optimizer_trace_figure():
    apply_style()
    case = representative_case()
    H = build_dense_hamiltonian(case.system, case.L)
    _, free_exact = exact_gibbs(H, case.beta)
    traces = {}
    final_rhos = {}
    for ansatz, seed in [("z_local", 7), ("xz_local", 7)]:
        rho, hist = variational_product_gibbs_with_history(H, case.beta, case.L, ansatz=ansatz, optimizer="BFGS", maxiter=180, init_scale=0.05, init_seed=seed)
        traces[ansatz] = [max(val - free_exact, 1e-12) for val in hist]
        final_rhos[ansatz] = rho

    lr2 = max(free_energy(low_rank_gibbs(H, case.beta, 2), H, case.beta) - free_exact, 1e-12)
    lr4 = max(free_energy(low_rank_gibbs(H, case.beta, 4), H, case.beta) - free_exact, 1e-12)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    color_map = {"z_local": inferno_colors(4)[1], "xz_local": inferno_colors(4)[3]}
    xs = np.arange(1, 1001)
    for ansatz, label in [("z_local", "z-local product ansatz"), ("xz_local", "xz-local product ansatz")]:
        vals = np.asarray(traces[ansatz], dtype=float)
        padded = np.full(1000, vals[-1])
        padded[: len(vals)] = vals
        ax.plot(xs, padded, color=color_map[ansatz], label=label)
        ax.scatter([len(vals)], [vals[-1]], color=color_map[ansatz], s=28, zorder=3)
    ax.hlines(lr2, 1, 1000, colors=inferno_colors(4)[0], linestyles="--", linewidth=1.8, label="Low-rank k=2")
    ax.hlines(lr4, 1, 1000, colors=inferno_colors(4)[2], linestyles=":", linewidth=2.2, label="Low-rank k=4")
    finish_axes(ax, xlabel="Optimization iteration budget", ylabel="Free-energy gap")
    ax.set_yscale("log")
    ax.set_xlim(-20, 1000)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "experiment3_gibbs_performance_vs_iterations.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_initial_vs_optimal_figure(cases: list[ThermalCase]):
    apply_style()
    labels = []
    initial_vals = []
    optimal_vals = []
    for case in cases:
        H = build_dense_hamiltonian(case.system, case.L)
        _, free_exact = exact_gibbs(H, case.beta)
        rho_initial = product_state_from_thetas(np.zeros(2 * case.L, dtype=float), case.L, ansatz="xz_local")
        rho_opt = variational_product_gibbs(H, case.beta, case.L, ansatz="xz_local", optimizer="BFGS", maxiter=180, init_scale=0.05, init_seed=7)
        short_system = "TFIM" if case.system == "tfim_critical" else "Heisenberg"
        labels.append(f"{short_system}\nL={case.L}, β={case.beta:g}")
        initial_vals.append(max(free_energy(rho_initial, H, case.beta) - free_exact, 1e-12))
        optimal_vals.append(max(free_energy(rho_opt, H, case.beta) - free_exact, 1e-12))

    x = np.arange(len(labels))
    width = 0.36
    colors = inferno_colors(2)
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x - width / 2, initial_vals, width=width, color=colors[0], label="initial product state")
    ax.bar(x + width / 2, optimal_vals, width=width, color=colors[1], label="optimized product state")
    ax.set_xticks(x, labels)
    ax.tick_params(axis='x', labelsize=9)
    finish_axes(ax, ylabel="Free-energy gap")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "experiment3_gibbs_initial_vs_optimal.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    cases = [
        ThermalCase("tfim_critical", 4, 0.5),
        ThermalCase("tfim_critical", 4, 1.0),
        ThermalCase("tfim_critical", 4, 2.0),
        ThermalCase("tfim_critical", 6, 0.5),
        ThermalCase("tfim_critical", 6, 1.0),
        ThermalCase("tfim_critical", 6, 2.0),
        ThermalCase("heisenberg_xxz", 4, 0.5),
        ThermalCase("heisenberg_xxz", 4, 1.0),
        ThermalCase("heisenberg_xxz", 4, 2.0),
    ]

    rows = []
    for case in cases:
        rows.extend(evaluate_case(case))

    payload = {
        "status": "complete",
        "experiment": "experiment3_gibbs_state_preparation",
        "rows": rows,
        "notes": [
            "Exact Gibbs states are computed by dense diagonalization on small spin chains.",
            "Two approximation families are benchmarked: low-rank spectral truncation and a variational product-state thermal ansatz.",
            "Metrics emphasize thermodynamic quality rather than circuit depth: free-energy gap, trace distance, fidelity, and observable error.",
        ],
    }
    (OUTDIR / "experiment3_gibbs_results.json").write_text(json.dumps(payload, indent=2))
    write_tsv(rows)
    (OUTDIR / "experiment3_gibbs_summary.json").write_text(json.dumps(summarize(rows), indent=2))
    make_main_figure(rows)
    make_optimizer_trace_figure()
    make_initial_vs_optimal_figure(cases)

    print("Wrote Gibbs benchmark artifacts to", OUTDIR)


if __name__ == "__main__":
    main()
