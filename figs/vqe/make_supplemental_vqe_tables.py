from __future__ import annotations

import ast
import csv
import difflib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = ROOT / "examples" / "vqe" / "snapshots"
OUTPUT = ROOT / "overleaf" / "generated_vqe_improvement_tables.tex"
ORDER = ["bh", "lih", "beh2", "h2o"]
MOLECULE_NAMES = {
    "bh": "BH",
    "lih": "LiH",
    "beh2": r"BeH\textsubscript{2}",
    "h2o": r"H\textsubscript{2}O",
}


@dataclass(frozen=True)
class Improvement:
    stem: str
    prev_iteration: int
    iteration: int
    prev_error: float
    error: float

    @property
    def gain_percent(self) -> float:
        return 100.0 * (1.0 - self.error / self.prev_error)


CORE_KEYS = ["ansatz", "param_model", "optimizer"]
ANSATZ_KEYS = ["ansatz", "param_model", "layers", "init_scale"]
OPTIMIZER_KEYS = ["optimizer", "max_steps", "step_size", "min_step_size", "seed", "cobyla_rhobeg", "cobyla_tol", "powell_xtol", "powell_ftol"]


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_key(key: str) -> str:
    return rf"\texttt{{{tex_escape(key)}}}"


def format_value(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return rf"\texttt{{{tex_escape(value[1:-1])}}}"
    return rf"\texttt{{{tex_escape(value)}}}"


def format_gain(value: float) -> str:
    if value >= 1.0:
        text = f"{value:.2f}"
    elif value >= 0.01:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text + r"\%"


def canonical_scalar(text: str):
    text = text.strip().rstrip(",")
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def scalar_equal(lhs, rhs) -> bool:
    if isinstance(lhs, float) or isinstance(rhs, float):
        try:
            return abs(float(lhs) - float(rhs)) <= 1e-15
        except Exception:
            return lhs == rhs
    return lhs == rhs


def value_to_tex(value) -> str:
    if isinstance(value, str):
        return rf"\texttt{{{tex_escape(value)}}}"
    return rf"\texttt{{{tex_escape(repr(value))}}}"


def extract_config_map(text: str) -> dict[str, object]:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("DEFAULT_CONFIG = RunConfig("):
            start = idx + 1
            break
    if start is None:
        raise RuntimeError("could not locate DEFAULT_CONFIG")

    result: dict[str, object] = {}
    idx = start
    while idx < len(lines):
        line = lines[idx].strip()
        if line == ")":
            break
        if not line:
            idx += 1
            continue
        if line == "initial_parameters=(),":
            result["initial_parameters"] = ()
            idx += 1
            continue
        if line.startswith("initial_parameters=("):
            values: list[float] = []
            idx += 1
            while idx < len(lines):
                inner = lines[idx].strip()
                if inner == "),":
                    break
                values.append(float(inner.rstrip(",")))
                idx += 1
            result["initial_parameters"] = tuple(values)
            idx += 1
            continue
        if "=" in line:
            key, raw = line.split("=", 1)
            result[key.strip()] = canonical_scalar(raw)
        idx += 1
    return result


def load_rows(stem: str) -> list[dict[str, str]]:
    path = SNAPSHOT_ROOT / stem / "results.tsv"
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row.get("iteration")]


def improvement_events(stem: str) -> list[Improvement]:
    rows = load_rows(stem)
    events: list[Improvement] = []
    best_iter: int | None = None
    best_error: float | None = None
    for row in rows:
        if row.get("status") != "keep" or not row.get("abs_final_error"):
            continue
        error = float(row["abs_final_error"])
        iteration = int(row["iteration"])
        if best_error is None:
            best_iter = iteration
            best_error = error
            continue
        if error < best_error:
            events.append(
                Improvement(
                    stem=stem,
                    prev_iteration=int(best_iter),
                    iteration=iteration,
                    prev_error=float(best_error),
                    error=error,
                )
            )
            best_iter = iteration
            best_error = error
    return events


def snapshot_file(stem: str, iteration: int) -> Path:
    root = SNAPSHOT_ROOT / stem / f"iter_{iteration:04d}"
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            archived_name = payload.get("archived_source_name")
            if archived_name:
                candidate = root / str(archived_name)
                if candidate.exists():
                    return candidate
    for name in ("simple_vqe.py",):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing snapshot source file in {root}")


def parse_assignment_changes(old_map: dict[str, object], new_map: dict[str, object]) -> tuple[list[str], list[str]]:
    ansatz_phrases: list[str] = []
    optimizer_phrases: list[str] = []
    for key in ANSATZ_KEYS + OPTIMIZER_KEYS:
        if key in old_map and key in new_map and not scalar_equal(old_map[key], new_map[key]):
            phrase = (
                f"{format_key(key)} {value_to_tex(old_map[key])} $\\rightarrow$ {value_to_tex(new_map[key])}"
            )
            if key in ANSATZ_KEYS:
                ansatz_phrases.append(phrase)
            else:
                optimizer_phrases.append(phrase)
    return ansatz_phrases, optimizer_phrases


def summarize_initial_parameters(old_map: dict[str, object], new_map: dict[str, object]) -> list[str]:
    old_values = tuple(old_map.get("initial_parameters", ()))
    new_values = tuple(new_map.get("initial_parameters", ()))
    if old_values == new_values:
        return []
    if not old_values and new_values:
        return [rf"replace the zero start with {len(new_values)} warm-start amplitudes"]
    if old_values and not new_values:
        return [r"drop the explicit warm-start amplitudes"]
    changed = sum(
        1
        for old_value, new_value in zip(old_values, new_values)
        if abs(float(old_value) - float(new_value)) > 1e-15
    )
    if len(old_values) != len(new_values):
        return [rf"resize the warm start from {len(old_values)} to {len(new_values)} amplitudes"]
    if changed == len(new_values):
        return [rf"retune all {len(new_values)} warm-start amplitudes"]
    return [rf"retune {changed} of {len(new_values)} warm-start amplitudes"]


def structural_phrases(old_text: str, new_text: str) -> tuple[list[str], list[str]]:
    ansatz_phrases: list[str] = []
    optimizer_phrases: list[str] = []

    if "def uccsd_kernel" in new_text and "def uccsd_kernel" not in old_text:
        ansatz_phrases.append(r"add a dedicated \texttt{uccsd} kernel")
    if "cudaq.kernels.uccsd" in new_text and "cudaq.kernels.uccsd" not in old_text:
        ansatz_phrases.append(r"switch the circuit body to CUDA-Q \texttt{uccsd}")
    if "optimizer_options" in new_text and "optimizer_options" not in old_text:
        optimizer_phrases.append(r"add explicit optimizer tolerance hooks")
    if "uccsd_spin_paired_symmetric" in new_text and "uccsd_spin_paired_symmetric" not in old_text:
        ansatz_phrases.append(r"add the \texttt{uccsd\_spin\_paired\_symmetric} map")
    if "REFERENCE_SPIN_PAIRED_FULL_DOUBLES" in new_text and "REFERENCE_SPIN_PAIRED_FULL_DOUBLES" not in old_text:
        ansatz_phrases.append(r"add the full-\texttt{uccsd} expansion map")
    if "run_coordinate_search" in old_text and "run_coordinate_search" in new_text:
        if "max_passes=max(4, 2 * len(result_x))" in old_text and "max_passes=max(4, 2 * len(result_x))" not in new_text:
            optimizer_phrases.append(r"remove the extra coordinate-search restart")
    if "def hea_ry_ring_kernel" in old_text and "def hea_ry_ring_kernel" not in new_text:
        ansatz_phrases.append(r"drop the HEA kernel path")

    def dedupe(items: list[str]) -> list[str]:
        deduped: list[str] = []
        for phrase in items:
            if phrase not in deduped:
                deduped.append(phrase)
        return deduped

    return dedupe(ansatz_phrases), dedupe(optimizer_phrases)


def combine_column(phrases: list[str]) -> str:
    if not phrases:
        return r"---"
    deduped: list[str] = []
    for phrase in phrases:
        if phrase not in deduped:
            deduped.append(phrase)
    return "; ".join(deduped[:5]) + "."


def summarize_diff_columns(stem: str, previous_iteration: int, current_iteration: int) -> tuple[str, str]:
    previous_path = snapshot_file(stem, previous_iteration)
    current_path = snapshot_file(stem, current_iteration)
    previous_text = previous_path.read_text()
    current_text = current_path.read_text()
    previous = previous_text.splitlines()
    current = current_text.splitlines()
    diff = list(difflib.unified_diff(previous, current, n=0))
    old_lines = [line for line in diff if line.startswith("-") and not line.startswith("---")]
    new_lines = [line for line in diff if line.startswith("+") and not line.startswith("+++")]
    old_map = extract_config_map(previous_text)
    new_map = extract_config_map(current_text)

    ansatz_phrases, optimizer_phrases = parse_assignment_changes(old_map, new_map)
    ansatz_phrases.extend(summarize_initial_parameters(old_map, new_map))
    structural_ansatz, structural_optimizer = structural_phrases(previous_text, current_text)
    ansatz_phrases.extend(structural_ansatz)
    optimizer_phrases.extend(structural_optimizer)

    if not ansatz_phrases and not optimizer_phrases:
        changed = []
        for line in old_lines[:2]:
            changed.append(rf"remove {format_value(line[1:].strip().rstrip(','))}")
        for line in new_lines[:2]:
            changed.append(rf"add {format_value(line[1:].strip().rstrip(','))}")
        ansatz_phrases.extend(changed)

    return combine_column(ansatz_phrases), combine_column(optimizer_phrases)


def zebra_row(cells: list[str], shaded: bool) -> str:
    widths = ["0.09\\linewidth", "0.15\\linewidth", "0.35\\linewidth", "0.35\\linewidth"]
    pieces = [
        rf"\parbox[c]{{{widths[0]}}}{{\centering {cells[0]}}}",
        rf"\parbox[c]{{{widths[1]}}}{{\centering {cells[1]}}}",
        rf"\parbox[c]{{{widths[2]}}}{{{cells[2]}}}",
        rf"\parbox[c]{{{widths[3]}}}{{{cells[3]}}}",
    ]
    inner = r"\hspace{0.01\linewidth}".join(pieces)
    if shaded:
        return rf"\noindent\colorbox{{black!4}}{{\parbox{{\dimexpr\linewidth-2\fboxsep\relax}}{{{inner}}}}}\par\smallskip"
    return rf"\noindent{inner}\par\smallskip"


def header_row(cells: list[str]) -> str:
    widths = ["0.09\\linewidth", "0.15\\linewidth", "0.35\\linewidth", "0.35\\linewidth"]
    pieces = [
        rf"\parbox[c][2.6em][c]{{{widths[0]}}}{{\centering {cells[0]}}}",
        rf"\parbox[c][2.6em][c]{{{widths[1]}}}{{\centering {cells[1]}}}",
        rf"\parbox[c][2.6em][c]{{{widths[2]}}}{{\centering {cells[2]}}}",
        rf"\parbox[c][2.6em][c]{{{widths[3]}}}{{\centering {cells[3]}}}",
    ]
    inner = r"\hspace{0.01\linewidth}".join(pieces)
    return rf"\noindent\colorbox{{black!4}}{{\parbox{{\dimexpr\linewidth-2\fboxsep\relax}}{{{inner}}}}}\par\smallskip"


def make_table(stem: str) -> str:
    molecule = MOLECULE_NAMES[stem]
    label = f"tab:supp_vqe_{stem}"
    caption = (
        rf"All improving {molecule} iterations. Gains are measured relative to the previous best archived "
        r"iteration in the same lane."
    )
    lines = [
        r"\refstepcounter{table}",
        rf"\label{{{label}}}",
        rf"\noindent\textbf{{Table \thetable.}} {caption}",
        r"\par\medskip",
        r"\noindent\rule{\linewidth}{0.5pt}",
        header_row(
            [
                r"\textbf{Iter.}",
                r"\textbf{Relative gain}",
                r"\textbf{Ansatz changes}",
                r"\textbf{Optimizer changes}",
            ]
        ),
        r"\noindent\rule{\linewidth}{0.5pt}",
    ]
    for idx, event in enumerate(improvement_events(stem)):
        ansatz_description, optimizer_description = summarize_diff_columns(
            stem, event.prev_iteration, event.iteration
        )
        lines.append(
            zebra_row(
                [
                    str(event.iteration),
                    format_gain(event.gain_percent),
                    ansatz_description,
                    optimizer_description,
                ],
                shaded=bool(idx % 2),
            )
        )
    lines.extend(
        [
            r"\noindent\rule{\linewidth}{0.5pt}",
            r"\par\medskip",
        ]
    )
    return "\n".join(lines)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for idx, stem in enumerate(ORDER):
        if idx:
            parts.append("")
            parts.append(r"\medskip")
            parts.append("")
        parts.append(make_table(stem))
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
