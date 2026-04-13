from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT.parent / "autoresearch" / "overleaf"
SUMMARY_OUTPUT = PAPER_ROOT / "generated_tn_summary_table.tex"
LOG_OUTPUT = PAPER_ROOT / "generated_tn_log_tables.tex"
RUNS = {
    "heisenberg_xxx_64": ROOT / "examples" / "tn" / "heisenberg_xxx_64" / "run_20260412_172834",
    "xxz_gapless_64": ROOT / "examples" / "tn" / "xxz_gapless_64" / "run_20260412_172834",
    "tfim_critical_64": ROOT / "examples" / "tn" / "tfim_critical_64" / "run_20260412_172834",
    "xx_critical_64": ROOT / "examples" / "tn" / "xx_critical_64" / "run_20260412_172834",
}
ORDER = [
    "heisenberg_xxx_64",
    "xxz_gapless_64",
    "tfim_critical_64",
    "xx_critical_64",
]
MODEL_NAMES = {
    "heisenberg_xxx_64": r"Heisenberg XXX, $L=64$",
    "xxz_gapless_64": r"Gapless XXZ, $L=64$",
    "tfim_critical_64": r"Critical TFIM, $L=64$",
    "xx_critical_64": r"Critical XX, $L=64$",
}
SUMMARY_MODEL_NAMES = {
    "heisenberg_xxx_64": r"Heisenberg XXX",
    "xxz_gapless_64": r"Gapless XXZ",
    "tfim_critical_64": r"Critical TFIM",
    "xx_critical_64": r"Critical XX",
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


def read_rows(stem: str) -> list[dict[str, object]]:
    path = RUNS[stem] / "logs" / "evaluations.jsonl"
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("type") != "evaluation":
            continue
        result = payload.get("result") or {}
        score = result.get("final_energy")
        if score is None:
            score = result.get("score")
        rows.append(
            {
                "iteration": int(payload["iteration"]),
                "description": str(payload.get("description", "")).strip(),
                "status": str(result.get("status", "")),
                "score": None if score is None else float(score),
                "config": result.get("config") or {},
            }
        )
    rows.sort(key=lambda row: int(row["iteration"]))
    return rows


def keep_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["status"] == "keep" and row["score"] is not None]


def best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(keep_rows(rows), key=lambda row: float(row["score"]))


def accept_reject_ratio(rows: list[dict[str, object]]) -> str:
    mutated = [row for row in rows if int(row["iteration"]) > 0]
    accepted = sum(1 for row in mutated if row["status"] == "keep")
    rejected = len(mutated) - accepted
    return f"{accepted}/{rejected}"


def format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e3:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def format_score(value: float | None) -> str:
    if value is None:
        return r"---"
    return f"{value:.6f}"


def format_relative_gain(previous_score: float | None, current_score: float | None) -> str:
    if previous_score is None or current_score is None or previous_score == 0.0:
        return r"---"
    gain = 100.0 * (previous_score - current_score) / abs(previous_score)
    if abs(gain) < 5e-5:
        gain = 0.0
    if abs(gain) >= 1.0:
        text = f"{gain:+.2f}"
    elif abs(gain) >= 0.01:
        text = f"{gain:+.4f}"
    else:
        text = f"{gain:+.6f}"
    text = text.rstrip("0").rstrip(".")
    if text in {"+0", "-0"}:
        text = "+0"
    return text + r"\%"


def summarize_method(cfg: dict[str, object]) -> str:
    phases = cfg.get("phases")
    if isinstance(phases, list) and phases:
        methods = r"$\rightarrow$".join(
            rf"\texttt{{{tex_escape(str(phase.get('method', '---')))}}}"
            for phase in phases
        )
        states = r"$\rightarrow$".join(
            rf"\texttt{{{tex_escape(str(phase.get('init_state', '---')))}}}"
            for phase in phases
        )
        return methods + "; init " + states + "."
    parts: list[str] = []
    method = cfg.get("method")
    if method:
        parts.append(rf"\texttt{{{tex_escape(str(method))}}}")
    init_state = cfg.get("init_state")
    if init_state:
        parts.append(rf"\texttt{{{tex_escape(str(init_state))}}}")
    init_bond_dim = cfg.get("init_bond_dim")
    if init_bond_dim not in (None, "", 0):
        parts.append(rf"init bond {init_bond_dim}")
    if not parts:
        return r"---"
    if len(parts) >= 2 and parts[1].startswith(r"\texttt{"):
        return parts[0] + "; init " + ", ".join(parts[1:]) + "."
    return "; ".join(parts) + "."


def summarize_solver(cfg: dict[str, object]) -> str:
    phases = cfg.get("phases")
    if isinstance(phases, list) and phases:
        summaries: list[str] = []
        for idx, phase in enumerate(phases, start=1):
            label = "warm" if idx == 1 else "refine" if idx == 2 else f"phase {idx}"
            bond_schedule = phase.get("bond_schedule") or []
            bond_text = "-".join(str(int(value)) for value in bond_schedule) if bond_schedule else "---"
            cutoff = format_number(float(phase["cutoff"])) if "cutoff" in phase else "---"
            solver_tol = format_number(float(phase["solver_tol"])) if "solver_tol" in phase else "---"
            max_sweeps = int(phase["max_sweeps"]) if "max_sweeps" in phase else "---"
            summaries.append(
                rf"{label}: bond {bond_text}; cutoff {cutoff}; tol {solver_tol}; sweeps {max_sweeps}"
            )
        return "; ".join(summaries[:2]) + "."
    parts: list[str] = []
    bond_schedule = cfg.get("bond_schedule")
    if isinstance(bond_schedule, list) and bond_schedule:
        bond_text = "-".join(str(int(value)) for value in bond_schedule)
        parts.append(rf"bond {bond_text}")
    for key in ("cutoff", "solver_tol", "max_sweeps", "local_eig_ncv"):
        if key in cfg:
            label = {
                "cutoff": "cutoff",
                "solver_tol": "tol",
                "max_sweeps": "sweeps",
                "local_eig_ncv": "ncv",
            }[key]
            value = cfg[key]
            if isinstance(value, (int, float)):
                parts.append(rf"{label} {format_number(float(value))}")
            else:
                parts.append(rf"{label} {tex_escape(str(value))}")
    if not parts:
        return r"---"
    return "; ".join(parts[:5]) + "."


def make_summary_table() -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Baseline and best archived tensor-network protocols for the four $L=64$ campaigns.}",
        r"\label{tab:supp_tn_protocols}",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\renewcommand{\tabularxcolumn}[1]{m{#1}}",
        r"\begin{tabularx}{\textwidth}{>{\centering\arraybackslash}m{0.105\textwidth} >{\centering\arraybackslash}m{0.07\textwidth} >{\centering\arraybackslash}m{0.095\textwidth} Y Y >{\centering\arraybackslash}m{0.11\textwidth}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{A/R} & \textbf{Protocol} & \textbf{Method / Initial State} & \textbf{Bond and Solver Settings} & \textbf{Final Energy} \\",
        r"\midrule",
    ]
    for idx, stem in enumerate(ORDER):
        rows = read_rows(stem)
        baseline = rows[0]
        best = best_row(rows)
        ratio = accept_reject_ratio(rows)
        lines.append(
            " & ".join(
                [
                    rf"\multirow[c]{{2}}{{=}}{{\centering {SUMMARY_MODEL_NAMES[stem]}}}",
                    rf"\multirow[c]{{2}}{{=}}{{\centering {ratio}}}",
                    "Initial",
                    summarize_method(baseline["config"]),
                    summarize_solver(baseline["config"]),
                    format_score(baseline["score"]),
                ]
            )
            + r" \\"
        )
        lines.append(r"\arrayrulecolor{black!28}\cmidrule(lr){3-6}\arrayrulecolor{black}")
        lines.append(
            " & ".join(
                [
                    "",
                    "",
                    rf"\shortstack[c]{{Best\\(Iter. {int(best['iteration'])})}}",
                    summarize_method(best["config"]),
                    summarize_solver(best["config"]),
                    format_score(best["score"]),
                ]
            )
            + r" \\"
        )
        if idx != len(ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
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
        rf"Tensor-network mutation log for {MODEL_NAMES[stem]}. Relative gain is measured "
        r"against the immediately previous scored iteration using the live final-energy objective."
    )
    lines = [
        r"\refstepcounter{table}",
        rf"\label{{tab:supp_tn_log_{stem}}}",
        rf"\noindent\textbf{{Table \thetable.}} {caption}",
        r"\par\medskip",
        r"\noindent\rule{\linewidth}{0.5pt}",
        log_header(
            [
                r"\textbf{Iter.}",
                r"\textbf{A/R}",
                r"\textbf{Relative gain}",
                r"\textbf{Score}",
                r"\textbf{Mutation summary}",
            ]
        ),
        r"\noindent\rule{\linewidth}{0.5pt}",
    ]

    previous_score: float | None = None
    for idx, row in enumerate(rows):
        status = str(row["status"])
        mark = CHECK_MARK if status == "keep" else CROSS_MARK
        score = row["score"]
        summary = tex_escape(str(row["description"]) or "no mutation summary recorded")
        lines.append(
            log_row(
                [
                    str(row["iteration"]),
                    mark,
                    format_relative_gain(previous_score, score),
                    format_score(score),
                    summary,
                ],
                shaded=(idx % 2 == 0),
            )
        )
        if score is not None:
            previous_score = float(score)

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
