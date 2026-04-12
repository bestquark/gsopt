from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from examples.afqmc.model_registry import ACTIVE_SYSTEMS as ACTIVE_AFQMC_SYSTEMS
from examples.dmrg.model_registry import ACTIVE_MODELS as ACTIVE_DMRG_MODELS
from examples.tn.model_registry import ACTIVE_MODELS as ACTIVE_TN_MODELS

from .common import benchmark_slug


@dataclass(frozen=True)
class LaneSpec:
    lane: str
    benchmark_arg: str
    source_filename: str
    queue_script: str | None
    restore_script: str | None
    plot_script: str
    snapshot_env: str
    fig_dir_env: str
    run_root_env: str | None
    optuna_root_env: str | None
    default_iterations: int
    default_wall_seconds: float
    objective_metric: str
    objective_text: str
    support_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExampleSpec:
    lane: LaneSpec
    example_key: str
    benchmark_value: str
    display_name: str
    source_template: str
    support_files: tuple[str, ...]

    def manifest_payload(self) -> dict:
        payload = {
            "version": 1,
            "lane": self.lane.lane,
            "example_key": self.example_key,
            "benchmark_arg": self.lane.benchmark_arg,
            "benchmark_value": self.benchmark_value,
            "display_name": self.display_name,
            "source_file": self.lane.source_filename,
            "evaluator_file": "evaluate.py",
            "source_template": self.source_template,
            "queue_script": self.lane.queue_script,
            "restore_script": self.lane.restore_script,
            "plot_script": self.lane.plot_script,
            "optuna_script": self.optuna_script,
            "snapshot_env": self.lane.snapshot_env,
            "fig_dir_env": self.lane.fig_dir_env,
            "run_root_env": self.lane.run_root_env,
            "optuna_root_env": self.lane.optuna_root_env,
            "default_iterations": self.lane.default_iterations,
            "default_wall_seconds": self.lane.default_wall_seconds,
            "objective_metric": self.lane.objective_metric,
            "objective_text": self.lane.objective_text,
            "support_files": list(self.support_files),
            "benchmark_storage_name": self.storage_name,
        }
        return payload

    @property
    def storage_name(self) -> str:
        if self.lane.benchmark_arg == "molecule":
            return benchmark_slug(self.benchmark_value)
        return self.benchmark_value

    @property
    def optuna_script(self) -> str:
        return str(Path(self.source_template).with_name("optuna_baseline.py"))


VQE_LANE = LaneSpec(
    lane="vqe",
    benchmark_arg="molecule",
    source_filename="simple_vqe.py",
    queue_script=None,
    restore_script=None,
    plot_script="figs/vqe/make_energy_figure.py",
    snapshot_env="AUTORESEARCH_VQE_SNAPSHOT_ROOT",
    fig_dir_env="AUTORESEARCH_VQE_FIG_DIR",
    run_root_env="AUTORESEARCH_VQE_RUN_ROOT",
    optuna_root_env="AUTORESEARCH_VQE_OPTUNA_ROOT",
    default_iterations=100,
    default_wall_seconds=20.0,
    objective_metric="final_energy",
    objective_text="Lower the final energy after exactly 20 seconds. Exact-energy error and chemical accuracy are offline comparison metrics, not the live score.",
)

TN_LANE = LaneSpec(
    lane="tn",
    benchmark_arg="model",
    source_filename="initial_script.py",
    queue_script=None,
    restore_script=None,
    plot_script="figs/tn/make_energy_figure.py",
    snapshot_env="AUTORESEARCH_TN_SNAPSHOT_ROOT",
    fig_dir_env="AUTORESEARCH_TN_FIG_DIR",
    run_root_env="AUTORESEARCH_TN_RUN_ROOT",
    optuna_root_env="AUTORESEARCH_TN_OPTUNA_ROOT",
    default_iterations=100,
    default_wall_seconds=20.0,
    objective_metric="final_energy",
    objective_text="Lower the final energy after exactly 20 seconds without changing the Hamiltonian or system size.",
    support_files=("examples/tn/model_registry.py", "examples/tn/reference_energies.py"),
)

AFQMC_LANE = LaneSpec(
    lane="afqmc",
    benchmark_arg="molecule",
    source_filename="initial_script.py",
    queue_script=None,
    restore_script=None,
    plot_script="figs/afqmc/make_energy_figure.py",
    snapshot_env="AUTORESEARCH_AFQMC_SNAPSHOT_ROOT",
    fig_dir_env="AUTORESEARCH_AFQMC_FIG_DIR",
    run_root_env=None,
    optuna_root_env="AUTORESEARCH_AFQMC_OPTUNA_ROOT",
    default_iterations=100,
    default_wall_seconds=300.0,
    objective_metric="final_energy_plus_2stderr",
    objective_text="Lower the 5-minute molecular AFQMC risk-adjusted score E + 2*stderr without changing the molecular geometry or basis. CCSD(T) error remains an offline comparison metric.",
    support_files=("examples/afqmc/model_registry.py", "examples/afqmc/reference_energies.py"),
)

DMRG_LANE = LaneSpec(
    lane="dmrg",
    benchmark_arg="model",
    source_filename="simple_dmrg.py",
    queue_script=None,
    restore_script=None,
    plot_script="figs/dmrg/make_energy_figure.py",
    snapshot_env="AUTORESEARCH_DMRG_SNAPSHOT_ROOT",
    fig_dir_env="AUTORESEARCH_DMRG_FIG_DIR",
    run_root_env="AUTORESEARCH_DMRG_RUN_ROOT",
    optuna_root_env=None,
    default_iterations=100,
    default_wall_seconds=20.0,
    objective_metric="final_energy",
    objective_text="Lower the final energy after exactly 20 seconds without changing the Hamiltonian or chain size. Excess energy is an offline comparison metric.",
)

LANE_SPECS = {
    spec.lane: spec
    for spec in (
        VQE_LANE,
        TN_LANE,
        AFQMC_LANE,
        DMRG_LANE,
    )
}


def _load_vqe_examples(repo_root: Path) -> list[ExampleSpec]:
    return [
        ExampleSpec(
            lane=VQE_LANE,
            example_key=candidate.parent.name,
            benchmark_value=candidate.parent.name.upper().replace("_PLUS", "+"),
            display_name=f"{candidate.parent.name.upper()} VQE",
            source_template=str(candidate.relative_to(repo_root)),
            support_files=(),
        )
        for candidate in sorted((repo_root / "examples" / "vqe").glob("*/simple_vqe.py"))
    ]


def _load_tn_examples() -> list[ExampleSpec]:
    return [
        ExampleSpec(
            lane=TN_LANE,
            example_key=model,
            benchmark_value=model,
            display_name=f"{model} TN",
            source_template=f"examples/tn/{model}/initial_script.py",
            support_files=TN_LANE.support_files,
        )
        for model in ACTIVE_TN_MODELS
    ]


def _load_afqmc_examples() -> list[ExampleSpec]:
    return [
        ExampleSpec(
            lane=AFQMC_LANE,
            example_key=system,
            benchmark_value=system,
            display_name=f"{system.upper()} Molecular AFQMC",
            source_template=f"examples/afqmc/{system}/initial_script.py",
            support_files=AFQMC_LANE.support_files,
        )
        for system in ACTIVE_AFQMC_SYSTEMS
    ]


def _load_dmrg_examples() -> list[ExampleSpec]:
    return [
        ExampleSpec(
            lane=DMRG_LANE,
            example_key=model,
            benchmark_value=model,
            display_name=f"{model} DMRG",
            source_template=f"examples/dmrg/{model}/simple_dmrg.py",
            support_files=(),
        )
        for model in ACTIVE_DMRG_MODELS
    ]


def load_examples(repo_root: Path) -> list[ExampleSpec]:
    return [
        *_load_vqe_examples(repo_root),
        *_load_tn_examples(),
        *_load_afqmc_examples(),
        *_load_dmrg_examples(),
    ]


def examples_by_source(repo_root: Path) -> dict[Path, ExampleSpec]:
    mapping: dict[Path, ExampleSpec] = {}
    for example in load_examples(repo_root):
        mapping[(repo_root / example.source_template).resolve()] = example
    return mapping


def examples_by_benchmark_root(repo_root: Path) -> dict[Path, ExampleSpec]:
    mapping: dict[Path, ExampleSpec] = {}
    for example in load_examples(repo_root):
        mapping[(repo_root / example.source_template).resolve().parent] = example
    return mapping


def asdict_example(example: ExampleSpec) -> dict:
    payload = asdict(example)
    payload["lane"] = example.lane.lane
    return payload
