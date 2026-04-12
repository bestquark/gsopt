from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style
from examples.tn.model_registry import MUTUAL_INFO_MODELS, PRETTY_LABELS
from examples.tn.reference_energies import reference_energy, reference_mutual_information
from examples.tn.simple_tn import mutual_information_matrix_from_mps_state

LANE_DIR = ROOT / "examples" / "tn"
FIG_DIR = Path(os.environ.get("AUTORESEARCH_TN_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "tn_mutual_information_error_overview.pdf"
OUTPUT_PNG = FIG_DIR / "tn_mutual_information_error_overview.png"
PLOT_ORDER = list(MUTUAL_INFO_MODELS)
WALL_SECONDS = 20.0
DISPLAY_TITLES = {
    "heisenberg_xxx_64": r"Heisenberg XXX",
    "xxz_gapless_64": r"Gapless XXZ",
    "tfim_critical_64": r"Critical TFIM",
    "xx_critical_64": r"Critical XX",
}


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _latest_run_dir(model: str) -> Path | None:
    benchmark_dir = LANE_DIR / model
    runs = sorted(path for path in benchmark_dir.glob("run_*") if path.is_dir())
    return runs[-1] if runs else None


def _snapshot_source(snapshot_dir: Path, fallback_name: str = "initial_script.py") -> Path:
    metadata_path = snapshot_dir / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        archived_name = payload.get("archived_source_name")
        if archived_name:
            candidate = snapshot_dir / str(archived_name)
            if candidate.exists():
                return candidate
    candidate = snapshot_dir / fallback_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"could not locate snapshot source in {snapshot_dir}")


def _run_modules_for_model(model: str):
    live_module = _load_module(LANE_DIR / model / "initial_script.py", f"_tn_mi_live_{model}")
    run_dir = _latest_run_dir(model)
    if run_dir is None:
        return {
            "baseline_module": live_module,
            "optimized_module": live_module,
            "best_iteration": None,
            "run_dir": None,
            "uses_run_snapshot": False,
        }

    best_path = run_dir / "best.json"
    baseline_snapshot = run_dir / "snapshots" / "iter_0000"
    if not best_path.exists() or not baseline_snapshot.exists():
        return {
            "baseline_module": live_module,
            "optimized_module": live_module,
            "best_iteration": None,
            "run_dir": run_dir,
            "uses_run_snapshot": False,
        }

    payload = json.loads(best_path.read_text())
    best_iteration = payload.get("iteration")
    source_snapshot = payload.get("source_snapshot")
    if best_iteration is None or source_snapshot is None:
        return {
            "baseline_module": live_module,
            "optimized_module": live_module,
            "best_iteration": None,
            "run_dir": run_dir,
            "uses_run_snapshot": False,
        }

    baseline_module = _load_module(
        _snapshot_source(baseline_snapshot),
        f"_tn_mi_base_{model}_{run_dir.name}",
    )
    optimized_module = _load_module(
        Path(source_snapshot),
        f"_tn_mi_best_{model}_{run_dir.name}_iter_{int(best_iteration):04d}",
    )
    return {
        "baseline_module": baseline_module,
        "optimized_module": optimized_module,
        "best_iteration": int(best_iteration),
        "run_dir": run_dir,
        "uses_run_snapshot": True,
    }


def _run_result(module, cfg) -> dict:
    problem = module.build_problem(module.MODEL_NAME)

    if hasattr(module, "run_strategy") and hasattr(module, "DEFAULT_STRATEGY"):
        start = time.perf_counter()
        state = None
        history: list[tuple[int, float, int]] = []
        phase_records: list[dict] = []
        for phase in module.DEFAULT_STRATEGY["phases"]:
            if (time.perf_counter() - start) >= WALL_SECONDS:
                break
            if phase["method"] in {"dmrg1", "dmrg2"}:
                state, phase_history = module.run_dmrg_phase(
                    phase,
                    problem,
                    state,
                    start_time=start,
                    wall_time_limit=WALL_SECONDS,
                )
            elif phase["method"] == "tebd1d":
                state, phase_history = module.run_tebd_phase(
                    phase,
                    problem,
                    state,
                    start_time=start,
                    wall_time_limit=WALL_SECONDS,
                )
            else:
                raise ValueError(f"unsupported method {phase['method']!r}")
            step_offset = len(history)
            for step, (energy, max_bond) in enumerate(phase_history, start=1):
                history.append((step_offset + step, energy, max_bond))
            phase_records.append(
                {
                    "name": phase["name"],
                    "method": phase["method"],
                    "steps": len(phase_history),
                    "final_energy": float(phase_history[-1][0]),
                    "max_bond_realized": int(phase_history[-1][1]),
                }
            )
        if not history:
            raise RuntimeError("strategy produced no optimization history")
        first_step, first_energy, _ = history[0]
        last_step, final_energy, final_max_bond = history[-1]
        result = {
            "config": module.DEFAULT_STRATEGY,
            "model": problem["name"],
            "geometry": problem["geometry"],
            "spin": problem["spin"],
            "nsites": problem["nsites"],
            "shape": [problem["lx"], problem["ly"]],
            "cyclic": problem["cyclic"],
            "wall_budget_seconds": WALL_SECONDS,
            "iterations": last_step,
            "final_energy": final_energy,
            "energy_per_site": final_energy / problem["nsites"],
            "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
            "wall_seconds": time.perf_counter() - start,
            "max_bond_realized": final_max_bond,
            "entropy_midchain": float(state.entropy(problem["lx"] // 2)) if state is not None else None,
            "mutual_information_matrix": None,
            "history": history,
            "phase_records": phase_records,
            "state_obj": None if state is None else state.copy(),
        }
    elif hasattr(module, "run_staged_dmrg") and hasattr(module, "run_dmrg_phase"):
        start = time.perf_counter()
        state = module.build_initial_mps(module.WARM_START_CONFIG, problem)
        history: list[tuple[int, float, int]] = []
        for phase_cfg in (module.WARM_START_CONFIG, cfg):
            state, phase_history = module.run_dmrg_phase(
                phase_cfg,
                problem,
                state,
                start_time=start,
                wall_time_limit=WALL_SECONDS,
            )
            offset = len(history)
            history.extend((offset + step, energy, max_bond) for step, energy, max_bond in phase_history)
        first_step, first_energy, _ = history[0]
        last_step, final_energy, final_max_bond = history[-1]
        result = {
            "config": {
                "name": "staged_dmrg1_then_dmrg2",
                "phases": [module.config_to_dict(module.WARM_START_CONFIG), module.config_to_dict(cfg)],
            },
            "model": problem["name"],
            "geometry": problem["geometry"],
            "spin": problem["spin"],
            "nsites": problem["nsites"],
            "shape": [problem["lx"], problem["ly"]],
            "cyclic": problem["cyclic"],
            "wall_budget_seconds": WALL_SECONDS,
            "iterations": last_step,
            "final_energy": final_energy,
            "energy_per_site": final_energy / problem["nsites"],
            "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
            "wall_seconds": time.perf_counter() - start,
            "max_bond_realized": final_max_bond,
            "entropy_midchain": float(state.entropy(problem["lx"] // 2)),
            "mutual_information_matrix": None,
            "history": history,
            "state_obj": state.copy(),
        }
    elif hasattr(module, "run_dmrg") and hasattr(module, "build_initial_mps") and hasattr(module, "qtn"):
        solver_cls = {"dmrg1": module.qtn.DMRG1, "dmrg2": module.qtn.DMRG2}[cfg.method]
        start = time.perf_counter()
        p0 = module.build_initial_mps(cfg, problem)
        solver = solver_cls(problem["mpo"], bond_dims=list(cfg.bond_schedule), cutoffs=cfg.cutoff, p0=p0)
        solver.opts["local_eig_tol"] = cfg.solver_tol
        solver.opts["local_eig_ncv"] = cfg.local_eig_ncv

        history: list[tuple[int, float, int]] = []
        previous_direction = None
        for sweep in range(cfg.max_sweeps):
            if (time.perf_counter() - start) >= WALL_SECONDS:
                break
            direction = "R" if sweep % 2 == 0 else "L"
            max_bond = cfg.bond_schedule[min(sweep, len(cfg.bond_schedule) - 1)]
            with module.warnings.catch_warnings():
                module.warnings.simplefilter("ignore")
                energy = solver.sweep(
                    direction=direction,
                    canonize=previous_direction in (None, direction),
                    max_bond=max_bond,
                    cutoff=cfg.cutoff,
                    cutoff_mode=solver.opts["bond_compress_cutoff_mode"],
                    method=solver.opts["bond_compress_method"],
                    verbosity=0,
                )
            energy = float(module.np.real(energy))
            solver.energies.append(energy)
            history.append((sweep + 1, energy, int(solver.state.max_bond())))
            previous_direction = direction

        if not history:
            energy = float(module.np.real((solver._b | solver.ham | solver._k) ^ all))
            history.append((1, energy, int(solver.state.max_bond())))

        first_step, first_energy, _ = history[0]
        last_step, final_energy, final_max_bond = history[-1]
        result = {
            "config": module.config_to_dict(cfg),
            "model": problem["name"],
            "geometry": problem["geometry"],
            "spin": problem["spin"],
            "nsites": problem["nsites"],
            "shape": [problem["lx"], problem["ly"]],
            "cyclic": problem["cyclic"],
            "wall_budget_seconds": WALL_SECONDS,
            "iterations": last_step,
            "final_energy": final_energy,
            "energy_per_site": final_energy / problem["nsites"],
            "energy_drop": first_energy - final_energy if first_step is not None else 0.0,
            "wall_seconds": time.perf_counter() - start,
            "max_bond_realized": final_max_bond,
            "entropy_midchain": float(solver.state.entropy(problem["lx"] // 2)) if problem["geometry"] == "1d" else None,
            "mutual_information_matrix": module.maybe_mutual_information_matrix(problem, solver.state),
            "history": history,
            "state_obj": solver.state.copy(),
        }
    else:
        run_config_sig = inspect.signature(module.run_config)
        kwargs = {"wall_time_limit": WALL_SECONDS}
        if "return_state" in run_config_sig.parameters:
            kwargs["return_state"] = True
        result = module.run_config(cfg, problem, **kwargs)

    state = result.pop("state_obj", None)
    if result.get("mutual_information_matrix") is None and problem["geometry"] == "1d" and state is not None:
        result["mutual_information_matrix"] = mutual_information_matrix_from_mps_state(state, problem["nsites"]).tolist()
    return result


def _error_matrix(result: dict, exact_matrix: np.ndarray) -> np.ndarray:
    approx = result.get("mutual_information_matrix")
    if approx is None:
        raise ValueError(f"missing mutual information matrix for {result.get('model')}")
    error = np.abs(np.asarray(approx, dtype=np.float64) - exact_matrix)
    np.fill_diagonal(error, np.nan)
    return error


def _tick_positions(nsites: int) -> tuple[list[int], list[str]]:
    indices = [0, nsites // 2, nsites - 1]
    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    return deduped, [str(index + 1) for index in deduped]


def _latex_sci(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.2f} \times 10^{{{exponent}}}"


def _panel_note(result: dict, ref_energy_value: float | None, error: np.ndarray) -> str:
    mean_abs_error = float(np.nanmean(error))
    if ref_energy_value is None:
        return rf"$\langle |\Delta I| \rangle = {_latex_sci(mean_abs_error)}$"
    energy_gap = float(result["final_energy"]) - ref_energy_value
    return (
        rf"$\Delta E = {_latex_sci(energy_gap)}$"
        + "\n"
        + rf"$\langle |\Delta I| \rangle = {_latex_sci(mean_abs_error)}$"
    )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    module_info = {model: _run_modules_for_model(model) for model in PLOT_ORDER}
    raw_exact_matrices = {model: reference_mutual_information(model) for model in PLOT_ORDER}
    missing = [model for model, matrix in raw_exact_matrices.items() if matrix is None]
    if missing:
        raise RuntimeError(f"missing exact mutual-information references for: {', '.join(missing)}")
    exact_matrices = {
        model: np.asarray(raw_exact_matrices[model], dtype=np.float64)
        for model in PLOT_ORDER
    }

    rows: list[list[tuple[str, dict, np.ndarray]]] = [[], []]
    all_positive_errors: list[float] = []
    reference_energies = {model: reference_energy(model) for model in PLOT_ORDER}

    for model in PLOT_ORDER:
        baseline_module = module_info[model]["baseline_module"]
        optimized_module = module_info[model]["optimized_module"]
        baseline_cfg = getattr(baseline_module, "BASELINE_CONFIG", None)
        if baseline_cfg is None:
            raise RuntimeError(f"{model} is missing BASELINE_CONFIG")
        optimized_cfg = getattr(optimized_module, "DEFAULT_CONFIG", None)
        if optimized_cfg is None and not hasattr(optimized_module, "DEFAULT_STRATEGY"):
            raise RuntimeError(f"{model} is missing DEFAULT_CONFIG/DEFAULT_STRATEGY")
        exact_matrix = exact_matrices[model]
        baseline_result = _run_result(baseline_module, baseline_cfg)
        optimized_result = _run_result(optimized_module, optimized_cfg)
        baseline_error = _error_matrix(baseline_result, exact_matrix)
        optimized_error = _error_matrix(optimized_result, exact_matrix)
        rows[0].append((model, baseline_result, baseline_error))
        rows[1].append((model, optimized_result, optimized_error))
        for matrix in (baseline_error, optimized_error):
            values = matrix[np.isfinite(matrix) & (matrix > 0.0)]
            all_positive_errors.extend(values.tolist())

    if not all_positive_errors:
        raise RuntimeError("no positive mutual-information errors were produced")

    vmin = max(min(all_positive_errors), 1e-8)
    vmax = max(all_positive_errors)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, len(PLOT_ORDER), figsize=(19.2, 9.2), sharex=True, sharey=True)
    if len(PLOT_ORDER) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)

    image = None
    for row_index, row in enumerate(rows):
        for col_index, (model, result, error) in enumerate(row):
            ax = axes[row_index, col_index]
            image = ax.imshow(error, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
            nsites = error.shape[0]
            ticks, ticklabels = _tick_positions(nsites)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            if row_index == 0:
                ax.set_title(DISPLAY_TITLES.get(model, PRETTY_LABELS[model]), pad=10)
                ax.tick_params(labelbottom=False)
            if row_index == 1:
                ax.set_xlabel(r"Site $j$")
            if col_index == 0:
                ax.set_ylabel(r"Site $i$")
            else:
                ax.tick_params(labelleft=False)
            ax.text(
                0.03,
                0.97,
                _panel_note(result, reference_energies[model], error),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=18,
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "boxstyle": "round,pad=0.28"},
            )

    fig.text(0.050, 0.74, "Initial", rotation=90, va="center", ha="center", fontsize=24)
    fig.text(0.050, 0.29, "Optimized", rotation=90, va="center", ha="center", fontsize=24)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.026, pad=0.065)
    colorbar.set_label(r"$|I_{\mathrm{TN}} - I_{\mathrm{ref}}|$", size=24)
    colorbar.ax.tick_params(labelsize=24)
    fig.subplots_adjust(left=0.11, right=0.85, bottom=0.12, top=0.90, wspace=0.20, hspace=0.05)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "overview_pdf": str(OUTPUT_PDF),
                "overview_png": str(OUTPUT_PNG),
                "models": PLOT_ORDER,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
