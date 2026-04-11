from __future__ import annotations

import ast
import csv
import difflib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = ROOT / "examples" / "tn" / "snapshots"
OUTPUT = ROOT / "overleaf" / "generated_tn_improvement_tables.tex"
ORDER = [
    "heisenberg_xxx_384",
    "xxz_gapless_256",
    "spin1_heisenberg_64",
    "tfim_2d_4x4",
    "heisenberg_2d_4x4",
]
MODEL_NAMES = {
    "heisenberg_xxx_384": r"Heisenberg XXX, $L=384$",
    "xxz_gapless_256": r"Gapless XXZ, $L=256$",
    "spin1_heisenberg_64": r"Spin-1 Heisenberg, $L=64$",
    "tfim_2d_4x4": r"TFIM, $4\times4$",
    "heisenberg_2d_4x4": r"Heisenberg, $4\times4$",
}


@dataclass(frozen=True)
class Improvement:
    stem: str
    prev_iteration: int
    iteration: int
    prev_energy: float
    energy: float
    final_best_energy: float

    @property
    def gain_percent(self) -> float:
        prev_gap = self.prev_energy - self.final_best_energy
        new_gap = self.energy - self.final_best_energy
        if prev_gap <= 1e-15:
            return 0.0
        return max(0.0, 100.0 * (1.0 - new_gap / prev_gap))


CONFIG_KEYS = [
    "method",
    "init_state",
    "bond_schedule",
    "cutoff",
    "solver_tol",
    "max_sweeps",
    "tau",
    "chi",
    "init_bond_dim",
    "init_seed",
    "local_eig_ncv",
]


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
    if isinstance(value, tuple):
        body = ", ".join(str(v) for v in value)
        return rf"\texttt{{({tex_escape(body)})}}"
    return rf"\texttt{{{tex_escape(repr(value))}}}"


def format_gain(value: float) -> str:
    if value >= 1.0:
        text = f"{value:.2f}"
    elif value >= 0.01:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text + r"\%"


def load_rows(stem: str) -> list[dict[str, str]]:
    path = SNAPSHOT_ROOT / stem / "results.tsv"
    with path.open() as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row.get("iteration")]


def snapshot_file(stem: str, iteration: int) -> Path:
    root = SNAPSHOT_ROOT / stem / f"iter_{iteration:04d}"
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text())
            archived_name = payload.get("archived_source_name")
            if archived_name:
                candidate = root / str(archived_name)
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass
    for name in ("initial_script.py",):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing snapshot source file in {root}")


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
        if "=" in line:
            key, raw = line.split("=", 1)
            result[key.strip()] = canonical_scalar(raw)
        idx += 1
    return result


def improvement_events(stem: str) -> list[Improvement]:
    rows = load_rows(stem)
    keep_rows = [
        (int(row["iteration"]), float(row["final_energy"]))
        for row in rows
        if row.get("status") == "keep" and row.get("final_energy")
    ]
    if not keep_rows:
        return []
    final_best_energy = min(energy for _, energy in keep_rows)

    events: list[Improvement] = []
    best_iter, best_energy = keep_rows[0]
    for iteration, energy in keep_rows[1:]:
        if energy < best_energy:
            events.append(
                Improvement(
                    stem=stem,
                    prev_iteration=best_iter,
                    iteration=iteration,
                    prev_energy=best_energy,
                    energy=energy,
                    final_best_energy=final_best_energy,
                )
            )
            best_iter, best_energy = iteration, energy
    return events


def config_phrases(old_map: dict[str, object], new_map: dict[str, object]) -> list[str]:
    phrases: list[str] = []
    for key in CONFIG_KEYS:
        if key in old_map and key in new_map and not scalar_equal(old_map[key], new_map[key]):
            phrases.append(
                f"{format_key(key)} {value_to_tex(old_map[key])} $\\rightarrow$ {value_to_tex(new_map[key])}"
            )
    return phrases


def structural_phrases(old_text: str, new_text: str) -> list[str]:
    phrases: list[str] = []
    checks = [
        ("np.random.seed(cfg.init_seed)" not in old_text and "np.random.seed(cfg.init_seed)" in new_text,
         r"seed the expansion noise with \texttt{init\_seed}"),
        ("rand_strength=0.01" in old_text and "rand_strength=1e-4" in new_text,
         r"shrink the MPS expansion noise $10^{-2} \rightarrow 10^{-4}$"),
        ("solver.solve(" in old_text and "solver.sweep(" in new_text,
         r"switch the DMRG inner update from \texttt{solve} to \texttt{sweep}"),
        ("bond_dims=[cfg.bond_schedule[0]]" in old_text and "bond_dims=list(cfg.bond_schedule)" in new_text,
         r"initialize DMRG with the full bond schedule"),
        ('"chain_length"' not in old_text and '"chain_length"' in new_text,
         r"log the chain length in the result payload"),
    ]
    for condition, phrase in checks:
        if condition:
            phrases.append(phrase)
    deduped: list[str] = []
    for phrase in phrases:
        if phrase not in deduped:
            deduped.append(phrase)
    return deduped


def fallback_phrases(old_text: str, new_text: str) -> list[str]:
    diff = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), n=0))
    old_lines = [line[1:].strip().rstrip(",") for line in diff if line.startswith("-") and not line.startswith("---")]
    new_lines = [line[1:].strip().rstrip(",") for line in diff if line.startswith("+") and not line.startswith("+++")]
    phrases: list[str] = []
    for line in old_lines[:2]:
        phrases.append(rf"remove \texttt{{{tex_escape(line)}}}")
    for line in new_lines[:2]:
        phrases.append(rf"add \texttt{{{tex_escape(line)}}}")
    return phrases


def summarize_diff(stem: str, previous_iteration: int, current_iteration: int) -> str:
    previous_text = snapshot_file(stem, previous_iteration).read_text()
    current_text = snapshot_file(stem, current_iteration).read_text()
    old_map = extract_config_map(previous_text)
    new_map = extract_config_map(current_text)

    phrases = config_phrases(old_map, new_map)
    phrases.extend(structural_phrases(previous_text, current_text))
    if not phrases:
        phrases.extend(fallback_phrases(previous_text, current_text))

    deduped: list[str] = []
    for phrase in phrases:
        if phrase not in deduped:
            deduped.append(phrase)
    if not deduped:
        return r"no code diff; rerun the same tensor-network policy."
    return "; ".join(deduped[:6]) + "."


def zebra_row(cells: list[str], shaded: bool) -> str:
    widths = ["0.10\\linewidth", "0.15\\linewidth", "0.70\\linewidth"]
    pieces = [
        rf"\parbox[c]{{{widths[0]}}}{{\centering {cells[0]}}}",
        rf"\parbox[c]{{{widths[1]}}}{{\centering {cells[1]}}}",
        rf"\parbox[c]{{{widths[2]}}}{{{cells[2]}}}",
    ]
    inner = r"\hspace{0.01\linewidth}".join(pieces)
    if shaded:
        return rf"\noindent\colorbox{{black!4}}{{\parbox{{\dimexpr\linewidth-2\fboxsep\relax}}{{{inner}}}}}\par\smallskip"
    return rf"\noindent{inner}\par\smallskip"


def header_row(cells: list[str]) -> str:
    widths = ["0.10\\linewidth", "0.15\\linewidth", "0.70\\linewidth"]
    pieces = [
        rf"\parbox[c][2.6em][c]{{{widths[0]}}}{{\centering {cells[0]}}}",
        rf"\parbox[c][2.6em][c]{{{widths[1]}}}{{\centering {cells[1]}}}",
        rf"\parbox[c][2.6em][c]{{{widths[2]}}}{{\centering {cells[2]}}}",
    ]
    inner = r"\hspace{0.01\linewidth}".join(pieces)
    return rf"\noindent\colorbox{{black!4}}{{\parbox{{\dimexpr\linewidth-2\fboxsep\relax}}{{{inner}}}}}\par\smallskip"


def make_table(stem: str) -> str:
    model = MODEL_NAMES[stem]
    label = f"tab:supp_tn_{stem}"
    caption = (
        rf"All improving {model} iterations. Gains are provisional reductions in the lane-relative "
        r"energy gap to the current best archived run in the same lane."
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
                r"\textbf{Description}",
            ]
        ),
        r"\noindent\rule{\linewidth}{0.5pt}",
    ]
    for idx, event in enumerate(improvement_events(stem)):
        lines.append(
            zebra_row(
                [
                    str(event.iteration),
                    format_gain(event.gain_percent),
                    summarize_diff(stem, event.prev_iteration, event.iteration),
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
    parts: list[str] = []
    for idx, stem in enumerate(ORDER):
        if idx:
            parts.extend(["", r"\medskip", ""])
        parts.append(make_table(stem))
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
