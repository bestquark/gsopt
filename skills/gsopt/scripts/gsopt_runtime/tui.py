from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from .runtime import RunContext, collect_status, locate_context


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    clean = " ".join(str(text).split())
    if len(clean) <= width:
        return clean
    if width <= 3:
        return clean[:width]
    return clean[: width - 3] + "..."


def _progress_bar(completed: int, target: int, width: int) -> str:
    width = max(4, width)
    if target <= 0:
        filled = 0
    else:
        filled = min(width, max(0, round(width * completed / target)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _campaign_state(context: RunContext) -> dict[str, Any] | None:
    return _read_json(context.logs_dir / "campaign" / "campaign_state.json")


def _slurm_state(context: RunContext) -> dict[str, Any] | None:
    return _read_json(context.logs_dir / "campaign" / "slurm" / "slurm_state.json")


def _rows_from_local_jsonl(context: RunContext) -> list[dict[str, Any]]:
    rows = []
    for event in _read_jsonl(context.evaluations_log):
        if event.get("type") != "evaluation":
            continue
        result = event.get("result") or {}
        rows.append(
            {
                "iteration": event.get("iteration"),
                "status": result.get("status", "-"),
                "score": result.get("score"),
                "description": event.get("description", ""),
                "timestamp": event.get("timestamp", ""),
            }
        )
    return rows


def _rows_from_results_tsv(context: RunContext) -> list[dict[str, Any]]:
    status = collect_status(context, write=False)
    results_path = Path(str(status.get("results_path", "")))
    if not results_path.exists() or results_path.suffix != ".tsv":
        return []
    rows: list[dict[str, Any]] = []
    metric_key = str(context.manifest.get("objective_metric", "score"))
    with results_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "iteration": row.get("iteration", ""),
                    "status": row.get("status", "-"),
                    "score": row.get(metric_key) or row.get("score"),
                    "description": row.get("description", ""),
                    "timestamp": row.get("timestamp", ""),
                }
            )
    return rows


def _recent_rows(context: RunContext, limit: int = 12) -> list[dict[str, Any]]:
    rows = _rows_from_local_jsonl(context)
    if not rows:
        rows = _rows_from_results_tsv(context)

    def key(row: dict[str, Any]) -> int:
        try:
            return int(row.get("iteration") or -1)
        except (TypeError, ValueError):
            return -1

    rows.sort(key=key)
    return rows[-limit:]


def _summary_lines(context: RunContext, width: int) -> list[str]:
    status = collect_status(context, write=context.is_run)
    campaign = _campaign_state(context)
    slurm = _slurm_state(context)
    completed = int(status.get("completed_mutations") or 0)
    target = int(status.get("target_iterations") or 0)
    bar_width = min(36, max(8, width - 42))
    done_text = "done" if status.get("done") else "running"
    lines = [
        f"GSOpt monitor | {done_text} | q quit | r refresh",
        _truncate(str(context.root_dir), width),
        f"{_progress_bar(completed, target, bar_width)} {completed}/{target} mutations, remaining {status.get('remaining_mutations')}",
        (
            f"lane={status.get('lane')} benchmark={status.get('benchmark_value')} "
            f"mode={status.get('evaluation_mode') or '-'} max_parallel={status.get('max_parallel') or '-'}"
        ),
        (
            f"latest={_fmt_value(status.get('latest_iteration'))} "
            f"best_iter={_fmt_value(status.get('best_iteration'))} "
            f"best_score={_fmt_value(status.get('best_metric'))}"
        ),
    ]
    if campaign:
        lines.append(
            "local campaign: "
            f"status={campaign.get('status', '-')} agent={campaign.get('agent', '-')} "
            f"launch={campaign.get('launch', campaign.get('launches', '-'))} "
            f"returncode={campaign.get('last_returncode', '-')}"
        )
    if slurm:
        lines.append(
            "slurm campaign: "
            f"status={slurm.get('status', '-')} agent={slurm.get('agent', '-')} "
            f"job={slurm.get('job_id', '-')} launches={slurm.get('launch_count', '-')} "
            f"no_progress={slurm.get('no_progress_count', '-')}"
        )
    lines.append("")
    lines.append("Recent evaluations")
    return [_truncate(line, width) for line in lines]


def _table_lines(context: RunContext, width: int, height: int) -> list[str]:
    rows = _recent_rows(context, limit=max(1, height - 9))
    if not rows:
        return ["No evaluations recorded yet."]
    iter_w = 6
    status_w = 9
    score_w = 14
    desc_w = max(12, width - iter_w - status_w - score_w - 7)
    header = f"{'iter':>{iter_w}}  {'status':<{status_w}}  {'score':>{score_w}}  description"
    sep = "-" * min(width, len(header))
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{_fmt_value(row.get('iteration')):>{iter_w}}  "
            f"{_truncate(_fmt_value(row.get('status')), status_w):<{status_w}}  "
            f"{_fmt_value(row.get('score')):>{score_w}}  "
            f"{_truncate(str(row.get('description') or ''), desc_w)}"
        )
    return [_truncate(line, width) for line in lines]


def render_text(path: Path, width: int = 100, height: int = 30) -> str:
    context = locate_context(path.resolve())
    lines = _summary_lines(context, width)
    lines.extend(_table_lines(context, width, max(8, height - len(lines))))
    return "\n".join(lines)


def _run_curses(path: Path, refresh_seconds: float) -> int:
    import curses

    context = locate_context(path.resolve())

    def draw(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(max(100, int(refresh_seconds * 1000)))
        while True:
            height, width = stdscr.getmaxyx()
            width = max(20, width - 1)
            height = max(8, height)
            stdscr.erase()
            lines = _summary_lines(context, width)
            lines.extend(_table_lines(context, width, height - len(lines)))
            for row_index, line in enumerate(lines[:height]):
                try:
                    stdscr.addstr(row_index, 0, line[:width])
                except curses.error:
                    pass
            stdscr.refresh()
            key = stdscr.getch()
            if key in {ord("q"), ord("Q"), 27}:
                return
            if key in {ord("r"), ord("R")}:
                continue

    curses.wrapper(draw)
    return 0


def run_tui(path: Path, refresh_seconds: float, once: bool = False) -> int:
    if once or not sys.stdout.isatty():
        print(render_text(path))
        return 0
    return _run_curses(path, refresh_seconds)
