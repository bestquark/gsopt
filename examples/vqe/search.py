from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simple_vqe import (
    CHEMICAL_ACCURACY,
    SUPPORTED_MOLECULES,
    RunConfig,
    build_problem,
    compact_result,
    config_from_dict,
    config_signature,
    config_to_dict,
    make_config_name,
    run_config,
)

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "bh.json"


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


def resolve_molecule_name(config: dict) -> str:
    molecule = config.get("molecule")
    if molecule is None:
        raise ValueError("VQE config must define a single `molecule`")
    if molecule not in SUPPORTED_MOLECULES:
        raise ValueError(f"unsupported molecule {molecule!r}; supported values: {SUPPORTED_MOLECULES}")
    return molecule


def random_config(search_space: dict, seed: int) -> RunConfig:
    rng = random.Random(seed)
    candidate = RunConfig(
        name="candidate",
        ansatz=rng.choice(search_space["ansatzes"]),
        layers=rng.choice(search_space["layers"]),
        optimizer=rng.choice(search_space["optimizers"]),
        max_steps=rng.choice(search_space["max_steps"]),
        init_scale=rng.choice(search_space["init_scale"]),
        seed=seed,
    )
    return config_from_dict({**config_to_dict(candidate), "name": make_config_name(candidate)})


def mutate_config(base: RunConfig, search_space: dict, seed: int) -> RunConfig:
    rng = random.Random(seed)
    data = config_to_dict(base)
    field = rng.choice(["ansatz", "layers", "optimizer", "max_steps", "init_scale", "seed"])
    if field == "ansatz":
        data[field] = rng.choice(search_space["ansatzes"])
    elif field == "layers":
        data[field] = rng.choice(search_space["layers"])
    elif field == "optimizer":
        data[field] = rng.choice(search_space["optimizers"])
    elif field == "max_steps":
        data[field] = rng.choice(search_space["max_steps"])
    elif field == "init_scale":
        data[field] = rng.choice(search_space["init_scale"])
    elif field == "seed":
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
    keep_top_k = search_cfg.get("keep_top_k", 25)
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
    raise RuntimeError("could not generate a fresh VQE configuration")


def make_record(iteration: int, candidate: RunConfig, result: dict) -> dict:
    compact = compact_result(result)
    return {
        "iteration": iteration,
        "status": "completed",
        "config": config_to_dict(candidate),
        "final_energy": compact["final_energy"],
        "final_error": compact["final_error"],
        "wall_seconds": compact["wall_seconds"],
        "result": compact,
    }


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    molecule_name = resolve_molecule_name(config)
    run_root = Path(config["logging"]["root"]) / config["run_name"]
    dirs = ensure_run_dirs(run_root)

    manifest_path = run_root / "manifest.json"
    iterations_jsonl = run_root / "iterations.jsonl"
    iterations_tsv = run_root / "iterations.tsv"
    best_path = run_root / "best_config.json"

    if not iterations_tsv.exists():
        iterations_tsv.write_text(
            "iteration\tconfig_name\tansatz\toptimizer\tlayers\tmax_steps\tfinal_energy\ttarget_energy\tfinal_error\tchem_acc_step\tcircuit_depth\twall_seconds\tseed\tstatus\n"
        )

    keep_top_k = config.get("search", {}).get("keep_top_k", 25)
    completed, seen, leaderboard = load_existing_state(iterations_jsonl, keep_top_k)
    problem = build_problem(molecule_name)

    manifest = {
        "lane": config.get("lane", "vqe"),
        "problem_label": config.get("problem_label"),
        "run_name": config["run_name"],
        "config_path": str(config_path),
        "iterations_target": config["iterations"],
        "candidate_wall_seconds": config["candidate_wall_seconds"],
        "molecule": molecule_name,
        "cas": list(problem["cas"]),
        "chemical_accuracy": config.get("chemical_accuracy", CHEMICAL_ACCURACY),
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
            chemical_accuracy=config.get("chemical_accuracy", CHEMICAL_ACCURACY),
            wall_time_limit=config["candidate_wall_seconds"],
        )

        history_path = dirs["histories"] / f"iter_{iteration:06d}_{molecule_name}.jsonl"
        append_text(
            history_path,
            "".join(
                json.dumps({"step": step, "energy": energy, "error": error}) + "\n"
                for step, energy, error in result["history"]
            ),
        )

        record = make_record(iteration, candidate, result)
        append_text(iterations_jsonl, json.dumps(record) + "\n")
        chem_acc_step = "" if result["chem_acc_step"] is None else result["chem_acc_step"]
        append_text(
            iterations_tsv,
            f"{iteration}\t{candidate.name}\t{candidate.ansatz}\t{candidate.optimizer}\t{candidate.layers}\t{candidate.max_steps}\t{result['final_energy']:.12f}\t{result['target_energy']:.12f}\t{result['final_error']:.6e}\t{chem_acc_step}\t{result['circuit_depth']}\t{result['wall_seconds']:.4f}\t{candidate.seed}\tcompleted\n",
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
            f"target_energy={result['target_energy']:.12f} final_error={result['final_error']:.6e} "
            f"chem_acc_step={result['chem_acc_step']} wall_seconds={result['wall_seconds']:.4f}"
        )


if __name__ == "__main__":
    main()
