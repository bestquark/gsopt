from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark import (
    ThermalCase,
    build_dense_hamiltonian,
    exact_gibbs,
    fidelity,
    free_energy,
    low_rank_gibbs,
    magnetization_z,
    trace_distance,
    variational_product_gibbs,
)

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "tfim6.json"

RANK_CHOICES = [2, 4, 6, 8]
OPTIMIZER_CHOICES = ["BFGS", "L-BFGS-B", "Nelder-Mead"]
MAXITER_CHOICES = [50, 100, 200, 400]
INIT_SCALE_CHOICES = [0.0, 0.05, 0.2, 1.0]
ANSATZ_CHOICES = ["z_local", "xz_local"]
METHOD_CHOICES = ["low_rank", "product"]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def append_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(text)


def signature(cfg: dict) -> tuple:
    return (
        cfg["method"],
        cfg.get("rank"),
        cfg.get("ansatz"),
        cfg.get("optimizer"),
        cfg.get("maxiter"),
        cfg.get("init_scale"),
        cfg.get("init_seed"),
    )


def random_config(rng: random.Random, iteration: int) -> dict:
    method = rng.choice(METHOD_CHOICES)
    base = {"name": f"gibbs_{iteration:04d}", "method": method}
    if method == "low_rank":
        base["rank"] = rng.choice(RANK_CHOICES)
    else:
        base["ansatz"] = rng.choice(ANSATZ_CHOICES)
        base["optimizer"] = rng.choice(OPTIMIZER_CHOICES)
        base["maxiter"] = rng.choice(MAXITER_CHOICES)
        base["init_scale"] = rng.choice(INIT_SCALE_CHOICES)
        base["init_seed"] = rng.randint(0, 1_000_000)
    return base


def mutate_config(base: dict, rng: random.Random, iteration: int) -> dict:
    cfg = dict(base)
    if cfg["method"] == "low_rank":
        if rng.random() < 0.2:
            cfg = random_config(rng, iteration)
        else:
            cfg["rank"] = rng.choice(RANK_CHOICES)
    else:
        field = rng.choice(["ansatz", "optimizer", "maxiter", "init_scale", "init_seed"])
        if field == "ansatz":
            cfg[field] = rng.choice(ANSATZ_CHOICES)
        elif field == "optimizer":
            cfg[field] = rng.choice(OPTIMIZER_CHOICES)
        elif field == "maxiter":
            cfg[field] = rng.choice(MAXITER_CHOICES)
        elif field == "init_scale":
            cfg[field] = rng.choice(INIT_SCALE_CHOICES)
        else:
            cfg[field] = rng.randint(0, 1_000_000)
    cfg["name"] = f"gibbs_{iteration:04d}"
    return cfg


def propose_config(cfg: dict, leaderboard: list[dict], seen: set[tuple], iteration: int) -> dict:
    rng = random.Random(cfg.get("seed", 42) + iteration * 1013)
    explore = cfg.get("search", {}).get("exploration_fraction", 0.25)
    for _ in range(1000):
        if leaderboard and rng.random() > explore:
            cand = mutate_config(rng.choice(leaderboard)["config"], rng, iteration)
        else:
            cand = random_config(rng, iteration)
        if signature(cand) not in seen:
            return cand
    raise RuntimeError("failed to propose fresh config")


def evaluate_candidate(candidate: dict, cases: list[ThermalCase]) -> tuple[list[dict], dict]:
    rows = []
    for case in cases:
        H = build_dense_hamiltonian(case.system, case.L)
        rho_exact, free_exact = exact_gibbs(H, case.beta)
        energy_exact = float(np.real(np.trace(rho_exact @ H)))
        mag_exact = magnetization_z(rho_exact, case.L)

        if candidate["method"] == "low_rank":
            t0 = time.perf_counter()
            rho = low_rank_gibbs(H, case.beta, candidate["rank"])
            runtime_s = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            rho = variational_product_gibbs(
                H,
                case.beta,
                case.L,
                ansatz=candidate["ansatz"],
                optimizer=candidate["optimizer"],
                maxiter=candidate["maxiter"],
                init_scale=candidate["init_scale"],
                init_seed=candidate["init_seed"],
            )
            runtime_s = time.perf_counter() - t0

        energy = float(np.real(np.trace(rho @ H)))
        free = free_energy(rho, H, case.beta)
        mag = magnetization_z(rho, case.L)
        rows.append(
            {
                "system": case.system,
                "L": case.L,
                "beta": case.beta,
                "energy_error": abs(energy - energy_exact),
                "free_energy_gap": free - free_exact,
                "trace_distance": trace_distance(rho, rho_exact),
                "fidelity": fidelity(rho, rho_exact),
                "magnetization_z_error": abs(mag - mag_exact),
                "runtime_s": runtime_s,
            }
        )

    aggregate = {
        "mean_energy_error": float(np.mean([r["energy_error"] for r in rows])),
        "mean_free_energy_gap": float(np.mean([r["free_energy_gap"] for r in rows])),
        "mean_trace_distance": float(np.mean([r["trace_distance"] for r in rows])),
        "mean_fidelity": float(np.mean([r["fidelity"] for r in rows])),
        "mean_magnetization_z_error": float(np.mean([r["magnetization_z_error"] for r in rows])),
        "mean_runtime_s": float(np.mean([r["runtime_s"] for r in rows])),
    }
    aggregate["score"] = [
        aggregate["mean_trace_distance"],
        aggregate["mean_free_energy_gap"],
        aggregate["mean_runtime_s"],
        -aggregate["mean_fidelity"],
        aggregate["mean_energy_error"],
    ]
    return rows, aggregate


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    run_root = Path(cfg["logging"]["root"]) / cfg["run_name"]
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    iterations_jsonl = run_root / "iterations.jsonl"
    iterations_tsv = run_root / "iterations.tsv"
    best_path = run_root / "best_config.json"

    if not iterations_tsv.exists():
        iterations_tsv.write_text(
            "iteration\tconfig_name\tmethod\trank\tansatz\toptimizer\tmaxiter\tinit_scale\tinit_seed\tmean_free_energy_gap\tmean_trace_distance\tmean_fidelity\tmean_energy_error\tmean_runtime_s\tstatus\n"
        )

    seen = set()
    leaderboard = []
    if iterations_jsonl.exists():
        for line in iterations_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "completed":
                continue
            seen.add(signature(record["config"]))
            leaderboard.append({"iteration": record["iteration"], "config": record["config"], "aggregate": record["aggregate"]})
    leaderboard = sorted(leaderboard, key=lambda r: tuple(r["aggregate"]["score"]))[: cfg.get("search", {}).get("keep_top_k", 25)]

    completed = 0 if not iterations_jsonl.exists() else len([x for x in iterations_jsonl.read_text().splitlines() if x.strip()])
    cases = [ThermalCase(item["system"], item["L"], item["beta"]) for item in cfg["cases"]]
    manifest = {
        "lane": cfg.get("lane", "gibbs"),
        "problem_label": cfg.get("problem_label"),
        "run_name": cfg["run_name"],
        "config_path": str(config_path),
        "iterations_target": cfg["iterations"],
        "cases": cfg["cases"],
        "objective": cfg.get("objective"),
        "resume": cfg["logging"].get("resume", True),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for iteration in range(completed + 1, cfg["iterations"] + 1):
        candidate = propose_config(cfg, leaderboard, seen, iteration)
        seen.add(signature(candidate))
        status = "completed"
        error_message = ""
        try:
            rows, aggregate = evaluate_candidate(candidate, cases)
        except Exception as exc:  # keep the search alive on numerical failures
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            rows = []
            aggregate = {
                "mean_energy_error": float("inf"),
                "mean_free_energy_gap": float("inf"),
                "mean_trace_distance": float("inf"),
                "mean_fidelity": 0.0,
                "mean_magnetization_z_error": float("inf"),
                "mean_runtime_s": float("inf"),
                "score": [float("inf"), float("inf"), float("inf"), 0.0, float("inf")],
            }

        record = {
            "iteration": iteration,
            "config": candidate,
            "rows": rows,
            "aggregate": aggregate,
            "status": status,
            "error": error_message,
        }
        append_text(iterations_jsonl, json.dumps(record) + "\n")
        append_text(
            iterations_tsv,
            f"{iteration}\t{candidate['name']}\t{candidate['method']}\t{candidate.get('rank','')}\t{candidate.get('ansatz','')}\t{candidate.get('optimizer','')}\t{candidate.get('maxiter','')}\t{candidate.get('init_scale','')}\t{candidate.get('init_seed','')}\t{aggregate['mean_free_energy_gap']:.6e}\t{aggregate['mean_trace_distance']:.6e}\t{aggregate['mean_fidelity']:.6f}\t{aggregate['mean_energy_error']:.6e}\t{aggregate['mean_runtime_s']:.6f}\t{status}\n",
        )
        if status == "completed":
            leaderboard.append({"iteration": iteration, "config": candidate, "aggregate": aggregate})
            leaderboard = sorted(leaderboard, key=lambda r: tuple(r["aggregate"]["score"]))[: cfg.get("search", {}).get("keep_top_k", 25)]
            best_path.write_text(json.dumps({"best_iteration": leaderboard[0]["iteration"], "best": leaderboard[0]}, indent=2))
        print(
            f"iter={iteration} status={status} method={candidate['method']} rank={candidate.get('rank')} ansatz={candidate.get('ansatz')} optimizer={candidate.get('optimizer')} maxiter={candidate.get('maxiter')} mean_free_gap={aggregate['mean_free_energy_gap']:.4e} mean_trace_dist={aggregate['mean_trace_distance']:.4f} mean_runtime={aggregate['mean_runtime_s']:.4f}s mean_fidelity={aggregate['mean_fidelity']:.4f}{' error=' + error_message if error_message else ''}"
        )


if __name__ == "__main__":
    main()
