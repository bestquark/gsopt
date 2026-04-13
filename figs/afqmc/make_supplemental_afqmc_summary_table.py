from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT.parent / "autoresearch" / "overleaf"
SUMMARY_OUTPUT = PAPER_ROOT / "generated_afqmc_summary_table.tex"

RUNS = {
    "h2": (ROOT / "examples" / "afqmc" / "h2" / "run_20260412_194332", None),
    "lih": (ROOT / "examples" / "afqmc" / "lih" / "run_20260412_194332", None),
    "h2o": (ROOT / "examples" / "afqmc" / "h2o" / "run_20260412_200020", None),
    "n2": (ROOT / "examples" / "afqmc" / "n2" / "run_20260412_204411", None),
}
ORDER = ["h2", "lih", "h2o", "n2"]
MOLECULE_NAMES = {
    "h2": r"H$_2$",
    "lih": "LiH",
    "h2o": r"H$_2$O",
    "n2": r"N$_2$",
}


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


def format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e3:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def read_rows(stem: str) -> list[dict[str, object]]:
    run_dir, limit = RUNS[stem]
    path = run_dir / "logs" / "evaluations.jsonl"
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("type") != "evaluation":
            continue
        iteration = int(payload["iteration"])
        if limit is not None and iteration > limit:
            continue
        result = payload.get("result") or {}
        score = result.get("score")
        rows.append(
            {
                "iteration": iteration,
                "status": str(result.get("status", "")),
                "score": None if score is None else float(score),
                "config": result.get("config") or {},
            }
        )
    rows.sort(key=lambda row: int(row["iteration"]))
    return rows


def best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    kept = [row for row in rows if row["status"] == "keep" and row["score"] is not None]
    return min(kept, key=lambda row: float(row["score"]))


def summarize_trial(cfg: dict[str, object]) -> str:
    parts: list[str] = []
    trial = cfg.get("trial")
    if trial:
        parts.append(rf"\texttt{{{tex_escape(str(trial))}}}")
    for key, label in (
        ("scf_conv_tol", "scf tol"),
        ("scf_max_cycle", "max cycle"),
        ("diis_space", "diis"),
    ):
        if key in cfg:
            parts.append(rf"{label} {format_number(float(cfg[key]))}")
    if not parts:
        return r"---"
    return "; ".join(parts[:4]) + "."


def summarize_propagation(cfg: dict[str, object]) -> str:
    parts: list[str] = []
    if "num_walkers_per_rank" in cfg:
        parts.append(rf"{int(cfg['num_walkers_per_rank'])} walkers/rank")
    if "num_steps_per_block" in cfg and "num_blocks" in cfg:
        parts.append(rf"$B={int(cfg['num_steps_per_block'])}\times {int(cfg['num_blocks'])}$")
    if "timestep" in cfg:
        parts.append(rf"$\Delta\tau={format_number(float(cfg['timestep']))}$")
    if "chol_cut" in cfg:
        parts.append(rf"chol cut {format_number(float(cfg['chol_cut']))}")
    if "stabilize_freq" in cfg and "pop_control_freq" in cfg:
        parts.append(rf"$s/p={int(cfg['stabilize_freq'])}/{int(cfg['pop_control_freq'])}$")
    if not parts:
        return r"---"
    return "; ".join(parts[:5]) + "."


def format_score(value: float | None) -> str:
    if value is None:
        return r"---"
    return f"{value:.6f}"


def stage_label(stem: str, row: dict[str, object], optimized: bool) -> str:
    if not optimized:
        return "Initial"
    return rf"Optimized ({int(row['iteration'])})"


def make_summary_table() -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Baseline and best archived AFQMC protocols for the four molecular campaigns.}",
        r"\label{tab:supp_afqmc_protocols}",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\renewcommand{\tabularxcolumn}[1]{m{#1}}",
        r"\begin{tabularx}{\textwidth}{>{\centering\arraybackslash}m{0.085\textwidth} >{\centering\arraybackslash}m{0.10\textwidth} Y Y >{\centering\arraybackslash}m{0.11\textwidth}}",
        r"\toprule",
        r"\textbf{Molecule} & \textbf{Protocol} & \textbf{Trial / SCF Settings} & \textbf{Walker and Propagation Settings} & \textbf{Live Score} \\",
        r"\midrule",
    ]
    for idx, stem in enumerate(ORDER):
        rows = read_rows(stem)
        baseline = rows[0]
        best = best_row(rows)
        lines.append(
            " & ".join(
                [
                    rf"\multirow[c]{{2}}{{=}}{{\centering {MOLECULE_NAMES[stem]}}}",
                    "Initial",
                    summarize_trial(baseline["config"]),
                    summarize_propagation(baseline["config"]),
                    format_score(baseline["score"]),
                ]
            )
            + r" \\"
        )
        lines.append(
            " & ".join(
                [
                    "",
                    rf"\shortstack[c]{{Best\\(Iter. {int(best['iteration'])})}}",
                    summarize_trial(best["config"]),
                    summarize_propagation(best["config"]),
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


def main():
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(make_summary_table())
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
