from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT.parent / "autoresearch" / "overleaf"
OUTPUT = PAPER_ROOT / "generated_tn_improvement_tables.tex"
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


@dataclass(frozen=True)
class Improvement:
    stem: str
    prev_iteration: int
    iteration: int
    prev_energy: float
    energy: float
    final_best_energy: float
    description: str

    @property
    def gain_percent(self) -> float:
        prev_gap = self.prev_energy - self.final_best_energy
        new_gap = self.energy - self.final_best_energy
        if prev_gap <= 1e-15:
            return 0.0
        return max(0.0, 100.0 * (1.0 - new_gap / prev_gap))


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


def format_gain(value: float) -> str:
    if value >= 1.0:
        text = f"{value:.2f}"
    elif value >= 0.01:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text + r"\%"


def run_dir(stem: str) -> Path:
    try:
        return RUNS[stem]
    except KeyError as exc:
        raise KeyError(f"no run directory configured for {stem!r}") from exc


def load_rows(stem: str) -> list[dict[str, object]]:
    path = run_dir(stem) / "logs" / "evaluations.jsonl"
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("type") != "evaluation":
            continue
        result = payload.get("result") or {}
        energy = result.get("final_energy")
        if energy is None:
            energy = result.get("score")
        rows.append(
            {
                "iteration": int(payload["iteration"]),
                "description": str(payload.get("description", "")).strip(),
                "status": result.get("status"),
                "final_energy": None if energy is None else float(energy),
            }
        )
    rows.sort(key=lambda row: int(row["iteration"]))
    return rows


def improvement_events(stem: str) -> list[Improvement]:
    rows = load_rows(stem)
    keep_rows = [
        (int(row["iteration"]), float(row["final_energy"]), str(row["description"]))
        for row in rows
        if row.get("status") == "keep" and row.get("final_energy") is not None
    ]
    if not keep_rows:
        return []

    final_best_energy = min(energy for _, energy, _ in keep_rows)
    events: list[Improvement] = []
    best_iter, best_energy, _ = keep_rows[0]
    for iteration, energy, description in keep_rows[1:]:
        if energy < best_energy - 1e-15:
            events.append(
                Improvement(
                    stem=stem,
                    prev_iteration=best_iter,
                    iteration=iteration,
                    prev_energy=best_energy,
                    energy=energy,
                    final_best_energy=final_best_energy,
                    description=description,
                )
            )
            best_iter, best_energy = iteration, energy
    return events


def describe_event(event: Improvement) -> str:
    text = event.description.strip()
    if not text:
        return "no mutation summary recorded."
    if not text.endswith("."):
        text += "."
    return tex_escape(text)


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
        rf"All improving {model} iterations. Gains are measured relative to the previous best archived "
        r"iteration in the same campaign."
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
                r"\textbf{Mutation summary}",
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
                    describe_event(event),
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
