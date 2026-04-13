from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT.parent / "autoresearch" / "overleaf"
OUTPUT = PAPER_ROOT / "generated_afqmc_improvement_tables.tex"

RUNS = [
    {
        "stem": "h2",
        "name": r"H$_2$",
        "run_dir": ROOT / "examples" / "afqmc" / "h2" / "run_20260412_194332",
        "limit": None,
        "label": "tab:supp_afqmc_h2",
        "caption_target": r"H$_2$ full 100-iteration campaign",
    },
    {
        "stem": "lih",
        "name": "LiH",
        "run_dir": ROOT / "examples" / "afqmc" / "lih" / "run_20260412_194332",
        "limit": None,
        "label": "tab:supp_afqmc_lih",
        "caption_target": r"LiH full 100-iteration campaign",
    },
    {
        "stem": "h2o",
        "name": r"H$_2$O",
        "run_dir": ROOT / "examples" / "afqmc" / "h2o" / "run_20260412_200020",
        "limit": 27,
        "label": "tab:supp_afqmc_h2o",
        "caption_target": r"H$_2$O rerun through iteration 27",
    },
    {
        "stem": "n2",
        "name": r"N$_2$",
        "run_dir": ROOT / "examples" / "afqmc" / "n2" / "run_20260412_204411",
        "limit": 5,
        "label": "tab:supp_afqmc_n2_v2",
        "caption_target": r"N$_2$ v2 rerun through iteration 5",
    },
]

CELL_WIDTHS = {
    "iter": "0.05",
    "ar": "0.04",
    "gain": "0.10",
    "score": "0.115",
    "summary": "0.671",
}
CELL_GAP = r"\hspace{0.006\linewidth}"
CHECK_MARK = r"\raisebox{0.08ex}{\ding{51}}"
CROSS_MARK = r"\raisebox{0.08ex}{\ding{55}}"


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


def normalize_description(text: str) -> str:
    text = re.sub(r"\s*=\s*", " = ", text.strip())
    text = re.sub(r",(?!\s)", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return tex_escape(text)


def read_rows(run_dir: Path, limit: int | None) -> list[dict[str, object]]:
    path = run_dir / "logs" / "evaluations.jsonl"
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("type") != "evaluation":
            continue
        iteration = int(payload["iteration"])
        if iteration == 0:
            continue
        if limit is not None and iteration > limit:
            continue
        result = payload.get("result") or {}
        score = result.get("score")
        rows.append(
            {
                "iteration": iteration,
                "status": str(result.get("status", "")),
                "score": None if score is None else float(score),
                "description": str(payload.get("description", "")),
            }
        )
    rows.sort(key=lambda row: int(row["iteration"]))
    return rows


def centered_cell(width: str, text: str, header: bool = False) -> str:
    if header:
        return rf"\parbox[c][2.2em][c]{{{width}\linewidth}}{{\centering {text}}}"
    return rf"\parbox[t]{{{width}\linewidth}}{{\centering\strut {text}}}"


def summary_cell(text: str) -> str:
    return rf"\parbox[t]{{{CELL_WIDTHS['summary']}\linewidth}}{{{text}}}"


def shaded_line(body: str) -> str:
    return (
        r"\noindent{\setlength{\fboxsep}{0pt}\colorbox{black!4}{\parbox{\linewidth}{"
        + body
        + r"}}}\par\smallskip"
    )


def plain_line(body: str) -> str:
    return r"\noindent" + body + r"\par\smallskip"


def format_score(score: float | None) -> str:
    if score is None:
        return r"---"
    return f"{score:.6f}"


def format_relative_gain(score: float | None, prev_scored: float | None, is_first_row: bool) -> str:
    if is_first_row or score is None or prev_scored is None or prev_scored == 0.0:
        return r"---"
    gain = (prev_scored - score) / abs(prev_scored) * 100.0
    return tex_escape(f"{gain:+.4f}%")


def status_mark(status: str) -> str:
    return CHECK_MARK if status == "keep" else CROSS_MARK


def render_header() -> str:
    body = CELL_GAP.join(
        [
            centered_cell(CELL_WIDTHS["iter"], r"\textbf{Iter.}", header=True),
            centered_cell(CELL_WIDTHS["ar"], r"\textbf{A/R}", header=True),
            centered_cell(CELL_WIDTHS["gain"], r"\textbf{Relative gain}", header=True),
            centered_cell(CELL_WIDTHS["score"], r"\textbf{Score}", header=True),
            centered_cell(CELL_WIDTHS["summary"], r"\textbf{Mutation summary}", header=True),
        ]
    )
    return shaded_line(body)


def render_row(row: dict[str, object], prev_scored: float | None, is_first_row: bool) -> tuple[str, float | None]:
    score = row["score"]
    body = CELL_GAP.join(
        [
            centered_cell(CELL_WIDTHS["iter"], str(int(row["iteration"]))),
            centered_cell(CELL_WIDTHS["ar"], status_mark(str(row["status"]))),
            centered_cell(CELL_WIDTHS["gain"], format_relative_gain(score, prev_scored, is_first_row)),
            centered_cell(CELL_WIDTHS["score"], format_score(score)),
            summary_cell(normalize_description(str(row["description"])) or r"---"),
        ]
    )
    line = shaded_line(body) if is_first_row or int(row["iteration"]) % 2 == 1 else plain_line(body)
    next_prev = prev_scored if score is None else float(score)
    return line, next_prev


def render_table(spec: dict[str, object]) -> str:
    rows = read_rows(Path(spec["run_dir"]), spec["limit"])
    lines = [
        r"\refstepcounter{table}",
        rf"\label{{{spec['label']}}}",
        (
            r"\noindent\textbf{Table \thetable.} AFQMC mutation log for "
            + str(spec["caption_target"])
            + r". Relative gain is measured against the immediately previous scored iteration using the live AFQMC score."
        ),
        r"\par\medskip",
        r"\noindent\rule{\linewidth}{0.5pt}",
        render_header(),
        r"\noindent\rule{\linewidth}{0.5pt}",
    ]
    prev_scored: float | None = None
    for index, row in enumerate(rows):
        line, prev_scored = render_row(row, prev_scored, is_first_row=(index == 0))
        lines.append(line)
    lines.extend(
        [
            r"\noindent\rule{\linewidth}{0.5pt}",
            r"\par\medskip",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(render_table(spec) for spec in RUNS) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
