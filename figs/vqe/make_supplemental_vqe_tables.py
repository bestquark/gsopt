from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT.parent / "autoresearch" / "overleaf"
LEGACY_ARCHIVES = [
    ROOT.parent / "autoresearch_legacy" / "gsopt_repo_archive_20260411_102814",
    ROOT.parent / "autoresearch_legacy" / "gsopt_repo_archive_20260411_102759",
]
LEGACY_ROOT = next((path for path in LEGACY_ARCHIVES if path.exists()), None)
if LEGACY_ROOT is None:
    raise FileNotFoundError("could not locate autoresearch_legacy VQE archive")

SNAPSHOT_ROOT = LEGACY_ROOT / "examples" / "vqe" / "snapshots"
SUMMARY_OUTPUT = PAPER_ROOT / "generated_vqe_summary_table.tex"
LOG_OUTPUT = PAPER_ROOT / "generated_vqe_log_tables.tex"
ORDER = ["bh", "lih", "beh2", "h2o"]
MOLECULE_NAMES = {
    "bh": "BH",
    "lih": "LiH",
    "beh2": r"BeH$_2$",
    "h2o": r"H$_2$O",
}

CHECK_MARK = r"\raisebox{0.08ex}{\ding{51}}"
CROSS_MARK = r"\raisebox{0.08ex}{\ding{55}}"
LOG_WIDTHS = [
    "0.05\\linewidth",
    "0.04\\linewidth",
    "0.10\\linewidth",
    "0.115\\linewidth",
    "0.671\\linewidth",
]
LOG_GAP = r"\hspace{0.006\linewidth}"


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


def read_rows(stem: str) -> list[dict[str, str]]:
    path = SNAPSHOT_ROOT / stem / "results.tsv"
    with path.open() as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row.get("iteration")]


def snapshot_file(stem: str, iteration: int) -> Path:
    return SNAPSHOT_ROOT / stem / f"iter_{iteration:04d}" / "simple_vqe.py"


def canonical_scalar(text: str):
    text = text.strip().rstrip(",")
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


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


def config_for_iteration(stem: str, iteration: int) -> dict[str, object]:
    text = snapshot_file(stem, iteration).read_text()
    config = extract_config_map(text)
    if "ansatz" not in config and ("cudaq.kernels.uccsd" in text or "def uccsd_kernel" in text):
        config["ansatz"] = "uccsd"
    return config


def row_error(row: dict[str, str]) -> float | None:
    value = row.get("abs_final_error")
    if not value:
        return None
    return abs(float(value))


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    kept = [row for row in rows if row.get("status") == "keep" and row_error(row) is not None]
    return min(kept, key=lambda row: float(row["abs_final_error"]))


def format_number(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    numeric = float(value)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    if 1e-3 <= magnitude < 1e3:
        text = f"{numeric:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return f"{numeric:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def format_error(value: float | None) -> str:
    if value is None:
        return r"---"
    if value == 0.0:
        return "0"
    exponent = int(f"{value:.1e}".split("e")[1])
    mantissa = value / (10 ** exponent)
    mantissa_text = f"{mantissa:.3f}".rstrip("0").rstrip(".")
    return rf"${mantissa_text}\times 10^{{{exponent}}}$"


def format_relative_gain(prev_error: float | None, current_error: float | None) -> str:
    if prev_error is None or current_error is None or prev_error <= 0.0:
        return r"---"
    gain = 100.0 * (prev_error - current_error) / prev_error
    if abs(gain) < 5e-5:
        gain = 0.0
    sign = "+" if gain >= 0 else ""
    magnitude = abs(gain)
    if magnitude >= 1.0:
        text = f"{gain:.2f}"
    elif magnitude >= 0.01:
        text = f"{gain:.4f}"
    else:
        text = f"{gain:.6f}"
    text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    if not text.startswith("-") and not text.startswith("+"):
        text = sign + text
    return text + r"\%"


def summarize_ansatz(cfg: dict[str, object]) -> str:
    parts: list[str] = []
    ansatz = cfg.get("ansatz")
    if ansatz:
        parts.append(rf"\texttt{{{tex_escape(str(ansatz))}}}")
    param_model = cfg.get("param_model")
    if param_model and param_model != ansatz:
        parts.append(rf"param. \texttt{{{tex_escape(str(param_model))}}}")
    layers = cfg.get("layers")
    if isinstance(layers, int) and layers:
        parts.append(rf"$L={layers}$")
    initial_parameters = cfg.get("initial_parameters", ())
    if isinstance(initial_parameters, tuple) and initial_parameters:
        parts.append(rf"warm start $n={len(initial_parameters)}$")
    if not parts:
        return r"---"
    return ", ".join(parts) + "."


def summarize_optimizer(cfg: dict[str, object]) -> str:
    parts: list[str] = []
    optimizer = cfg.get("optimizer")
    if optimizer:
        parts.append(rf"\texttt{{{tex_escape(str(optimizer))}}}")
    if "max_steps" in cfg:
        parts.append(rf"\texttt{{max\_steps}}={cfg['max_steps']}")
    if "init_scale" in cfg:
        parts.append(rf"\texttt{{init\_scale}}={format_number(cfg['init_scale'])}")
    if "seed" in cfg:
        parts.append(rf"seed {cfg['seed']}")
    if optimizer == "cobyla" and "cobyla_rhobeg" in cfg:
        parts.append(rf"\texttt{{rhobeg}}={format_number(cfg['cobyla_rhobeg'])}")
    if optimizer == "powell" and "powell_xtol" in cfg:
        parts.append(rf"\texttt{{xtol}}={format_number(cfg['powell_xtol'])}")
    if not parts:
        return r"---"
    return ", ".join(parts[:5]) + "."


def make_summary_table() -> str:
    lines = [
        r"\begin{table}[h]",
        r"\caption{Initial and best archived VQE protocols for the four completed molecular campaigns used in the supplement.}",
        r"\label{tab:supp_vqe_protocols}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c l l c}",
        r"\toprule",
        r"\textbf{Molecule} & \textbf{Stage} & \textbf{Ansatz / Parameterization} & \textbf{Optimizer and Key Settings} & \textbf{$|\Delta E|$ [Ha]} \\",
        r"\midrule",
    ]
    for stem in ORDER:
        rows = read_rows(stem)
        baseline = rows[0]
        best = best_row(rows)
        for stage, row in (
            ("Initial", baseline),
            (rf"Optimized ({int(best['iteration'])})", best),
        ):
            cfg = config_for_iteration(stem, int(row["iteration"]))
            lines.append(
                " & ".join(
                    [
                        MOLECULE_NAMES[stem],
                        stage,
                        summarize_ansatz(cfg),
                        summarize_optimizer(cfg),
                        format_error(row_error(row)),
                    ]
                )
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def log_cell(width: str, text: str, center: bool = False, header: bool = False) -> str:
    if header:
        return rf"\parbox[c][2.2em][c]{{{width}}}{{\centering {text}}}"
    if center:
        return rf"\parbox[t]{{{width}}}{{\centering\strut {text}}}"
    return rf"\parbox[t]{{{width}}}{{{text}}}"


def log_header(cells: list[str]) -> str:
    inner = LOG_GAP.join(log_cell(width, cell, header=True) for width, cell in zip(LOG_WIDTHS, cells))
    return (
        r"\noindent{\setlength{\fboxsep}{0pt}\colorbox{black!4}{\parbox{\linewidth}{"
        + inner
        + r"}}}\par\smallskip"
    )


def log_row(cells: list[str], shaded: bool) -> str:
    inner = LOG_GAP.join(
        [
            log_cell(LOG_WIDTHS[0], cells[0], center=True),
            log_cell(LOG_WIDTHS[1], cells[1], center=True),
            log_cell(LOG_WIDTHS[2], cells[2], center=True),
            log_cell(LOG_WIDTHS[3], cells[3], center=True),
            log_cell(LOG_WIDTHS[4], cells[4]),
        ]
    )
    if shaded:
        return (
            r"\noindent{\setlength{\fboxsep}{0pt}\colorbox{black!4}{\parbox{\linewidth}{"
            + inner
            + r"}}}\par\smallskip"
        )
    return rf"\noindent{inner}\par\smallskip"


def make_log_table(stem: str) -> str:
    rows = read_rows(stem)[1:]
    caption = (
        rf"VQE mutation log for {MOLECULE_NAMES[stem]} full 100-iteration archival campaign. "
        r"Relative gain is measured against the immediately previous scored iteration using the absolute energy error."
    )
    lines = [
        r"\refstepcounter{table}",
        rf"\label{{tab:supp_vqe_log_{stem}}}",
        rf"\noindent\textbf{{Table \thetable.}} {caption}",
        r"\par\medskip",
        r"\noindent\rule{\linewidth}{0.5pt}",
        log_header(
            [
                r"\textbf{Iter.}",
                r"\textbf{A/R}",
                r"\textbf{Relative gain}",
                r"\textbf{$|\Delta E|$ [Ha]}",
                r"\textbf{Mutation summary}",
            ]
        ),
        r"\noindent\rule{\linewidth}{0.5pt}",
    ]

    previous_error: float | None = None
    for idx, row in enumerate(rows):
        current_error = row_error(row)
        status = row.get("status", "")
        mark = CHECK_MARK if status == "keep" else CROSS_MARK
        summary = tex_escape((row.get("description") or "").strip() or "no mutation summary recorded")
        lines.append(
            log_row(
                [
                    row["iteration"],
                    mark,
                    format_relative_gain(previous_error, current_error),
                    format_error(current_error),
                    summary,
                ],
                shaded=(idx % 2 == 0),
            )
        )
        if current_error is not None:
            previous_error = current_error

    lines.extend(
        [
            r"\noindent\rule{\linewidth}{0.5pt}",
            r"\par\medskip",
        ]
    )
    return "\n".join(lines)


def make_log_tables() -> str:
    parts: list[str] = []
    for idx, stem in enumerate(ORDER):
        if idx:
            parts.extend(["", r"\medskip", ""])
        parts.append(make_log_table(stem))
    return "\n".join(parts) + "\n"


def main():
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(make_summary_table())
    LOG_OUTPUT.write_text(make_log_tables())
    print(SUMMARY_OUTPUT)
    print(LOG_OUTPUT)


if __name__ == "__main__":
    main()
