from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simple_dmrg import (
    SUPPORTED_MODELS,
    RunConfig,
    build_problem,
    compact_result,
    config_from_dict,
    config_signature,
    config_to_dict,
    make_config_name,
    run_config,
)

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "heisenberg_xxx_64.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def append_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)


def ensure_run_dirs(root: Path) -> dict:
    paths = {
        "root": root,
        "histories": root / "histories",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_model_name(config: dict) -> str:
    model = config.get("model")
    if model is None:
        raise ValueError("DMRG config must define a single `model`")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model!r}; supported values: {SUPPORTED_MODELS}")
    return model


def random_config(search_space: dict, seed: int) -> RunConfig:
    rng = random.Random(seed)
    candidate = RunConfig(
        name="candidate",
        bond_schedule=tuple(rng.choice(search_space["bond_schedules"])),
        cutoff=rng.choice(search_space["cutoffs"]),
        solver_tol=rng.choice(search_space["solver_tols"]),
        max_sweeps=rng.choice(search_space["max_sweeps"]),
        init_bond_dim=rng.choice(search_space["init_bond_dims"]),
        init_seed=seed,
    )
    return config_from_dict({**config_to_dict(candidate), "name": make_config_name(candidate)})


def mutate_config(base: RunConfig, search_space: dict, seed: int) -> RunConfig:
    rng = random.Random(seed)
    data = config_to_dict(base)
    field = rng.choice(["bond_schedule", "cutoff", "solver_tol", "max_sweeps", "init_bond_dim", "init_seed"])
    if field == "bond_schedule":
        data[field] = rng.choice(search_space["bond_schedules"])
    elif field == "cutoff":
        data[field] = rng.choice(search_space["cutoffs"])
    elif field == "solver_tol":
        data[field] = rng.choice(search_space["solver_tols"])
    elif field == "max_sweeps":
        data[field] = rng.choice(search_space["max_sweeps"])
    elif field == "init_bond_dim":
        data[field] = rng.choice(search_space["init_bond_dims"])
    elif field == "init_seed":
        data[field] = seed
    candidate = config_from_dict(data)
    return config_from_dict({**config_to_dict(candidate), "name": make_config_name(candidate)})


def seed_config_for_iteration(config: dict, iteration: int) -> RunConfig | None:
    seed_configs = config.get("seed_configs", [])
    if 1 <= iteration <= len(seed_configs):
        seeded = config_from_dict(seed_configs[iteration - 1])
        return config_from_dict({**config_to_dict(seeded), "name": make_config_name(seeded)})
    return None


def score_from_record(record: dict) -> tuple[float]:
    return (record.get("final_energy", float("inf")),)


def load_existing_state(iterations_jsonl: Path, keep_top_k: int) -> tuple[int, set[tuple], list[dict]]:
    completed = 0
    seen: set[tuple] = set()
    leaderboard: list[dict] = []
    if not iterations_jsonl.exists():
        return completed, seen, leaderboard

    with iterations_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            completed += 1
            record = json.loads(line)
            cfg = config_from_dict(record["config"])
            seen.add(config_signature(cfg))
            if record.get("status") != "completed":
                continue
            leaderboard.append(
                {
                    "score": score_from_record(record),
                    "iteration": record["iteration"],
                    "config": record["config"],
                    "config_obj": cfg,
                    "result": record.get("result"),
                }
            )
    leaderboard.sort(key=lambda item: item["score"])
    return completed, seen, leaderboard[:keep_top_k]


def propose_config(config: dict, seen: set[tuple], leaderboard: list[dict], iteration: int) -> RunConfig:
    seeded = seed_config_for_iteration(config, iteration)
    if seeded is not None and config_signature(seeded) not in seen:
        return seeded

    search_cfg = config.get("search", {})
    search_space = config["search_space"]
    keep_top_k = search_cfg.get("keep_top_k", 20)
    exploration_fraction = search_cfg.get("exploration_fraction", 0.25)
    proposal_policy = search_cfg.get("proposal_policy", "mutate_best_plus_random")

    for offset in range(10_000):
        candidate_seed = config.get("seed", 42) + iteration * 1000 + offset
        if proposal_policy == "mutate_best_plus_random" and leaderboard and (
            offset > 0 or (iteration % 10) / 10.0 >= exploration_fraction
        ):
            base = leaderboard[min(len(leaderboard) - 1, iteration % max(1, min(len(leaderboard), keep_top_k)))]
            candidate = mutate_config(base["config_obj"], search_space, candidate_seed)
        else:
            candidate = random_config(search_space, candidate_seed)
        if config_signature(candidate) not in seen:
            return candidate
    raise RuntimeError("could not generate a fresh DMRG configuration")


def make_record(iteration: int, candidate: RunConfig, result: dict) -> dict:
    compact = compact_result(result)
    return {
        "iteration": iteration,
        "status": "completed",
        "config": config_to_dict(candidate),
        "final_energy": compact["final_energy"],
        "energy_per_site": compact["energy_per_site"],
        "energy_drop": compact["energy_drop"],
        "wall_seconds": compact["wall_seconds"],
        "result": compact,
    }


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    model_name = resolve_model_name(config)
    run_root = Path(config["logging"]["root"]) / config["run_name"]
    dirs = ensure_run_dirs(run_root)

    manifest_path = run_root / "manifest.json"
    iterations_jsonl = run_root / "iterations.jsonl"
    iterations_tsv = run_root / "iterations.tsv"
    best_path = run_root / "best_config.json"

    if not iterations_tsv.exists():
        iterations_tsv.write_text(
            "iteration\tconfig_name\tbond_schedule\tcutoff\tsolver_tol\tmax_sweeps\tinit_bond_dim\tfinal_energy\tenergy_per_site\tenergy_drop\tmax_bond_realized\twall_seconds\tseed\tstatus\n"
        )

    keep_top_k = config.get("search", {}).get("keep_top_k", 20)
    completed, seen, leaderboard = load_existing_state(iterations_jsonl, keep_top_k)
    problem = build_problem(model_name)

    manifest = {
        "lane": config.get("lane", "dmrg"),
        "problem_label": config.get("problem_label"),
        "run_name": config["run_name"],
        "config_path": str(config_path),
        "iterations_target": config["iterations"],
        "candidate_wall_seconds": config["candidate_wall_seconds"],
        "model": model_name,
        "objective": config.get("objective"),
        "resume": config["logging"].get("resume", True),
        "seed_configs": config.get("seed_configs", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for iteration in range(completed + 1, config["iterations"] + 1):
        candidate = propose_config(config, seen, leaderboard, iteration)
        seen.add(config_signature(candidate))
        result = run_config(
            candidate,
            problem,
            wall_time_limit=config["candidate_wall_seconds"],
        )

        history_path = dirs["histories"] / f"iter_{iteration:06d}_{model_name}.jsonl"
        append_text(
            history_path,
            "".join(
                json.dumps({"step": step, "energy": energy, "max_bond": max_bond}) + "\n"
                for step, energy, max_bond in result["history"]
            ),
        )

        record = make_record(iteration, candidate, result)
        append_text(iterations_jsonl, json.dumps(record) + "\n")
        append_text(
            iterations_tsv,
            f"{iteration}\t{candidate.name}\t{','.join(map(str, candidate.bond_schedule))}\t{candidate.cutoff:.0e}\t{candidate.solver_tol:.0e}\t{candidate.max_sweeps}\t{candidate.init_bond_dim}\t{result['final_energy']:.12f}\t{result['energy_per_site']:.8f}\t{result['energy_drop']:.6e}\t{result['max_bond_realized']}\t{result['wall_seconds']:.4f}\t{candidate.init_seed}\tcompleted\n",
        )

        entry = {
            "score": score_from_record(record),
            "iteration": iteration,
            "config": config_to_dict(candidate),
            "config_obj": candidate,
            "result": record["result"],
        }
        leaderboard.append(entry)
        leaderboard.sort(key=lambda item: item["score"])
        leaderboard = leaderboard[:keep_top_k]

        incumbent = leaderboard[0]
        best_path.write_text(
            json.dumps(
                {
                    "iteration": incumbent["iteration"],
                    "config": incumbent["config"],
                    "result": incumbent["result"],
                },
                indent=2,
            )
        )
        print(
            f"iter={iteration} config={candidate.name} final_energy={result['final_energy']:.12f} "
            f"energy_per_site={result['energy_per_site']:.8f} energy_drop={result['energy_drop']:.6e} "
            f"max_bond={result['max_bond_realized']} wall_seconds={result['wall_seconds']:.4f}"
        )


if __name__ == "__main__":
    main()
