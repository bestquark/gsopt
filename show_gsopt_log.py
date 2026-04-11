from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_run_and_log(path_arg: str) -> tuple[Path, Path]:
    start = Path(path_arg).resolve()

    if start.is_file():
        if start.name != "evaluations.jsonl":
            raise SystemExit(f"expected an evaluations.jsonl file, got {start}")
        return start.parent.parent, start

    for candidate in (start, *start.parents):
        log_path = candidate / "logs" / "evaluations.jsonl"
        if log_path.exists():
            return candidate, log_path

    latest_run = _latest_run_dir(start)
    if latest_run is not None:
        return latest_run, latest_run / "logs" / "evaluations.jsonl"

    raise SystemExit(f"could not find a GSOpt run under {start}")


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    runs = sorted(
        path
        for path in root.glob("run_*")
        if path.is_dir() and (path / "logs" / "evaluations.jsonl").exists()
    )
    if runs:
        return runs[-1]
    return None


def _load_rows(log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("type") == "evaluation":
            rows.append(payload)
    return rows


def _format_score(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.12e}"
    except (TypeError, ValueError):
        return str(value)


def _truncate(text: str, width: int) -> str:
    if width < 4 or len(text) <= width:
        return text
    return text[: width - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact GSOpt evaluation table.")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Run dir, benchmark dir, or logs/evaluations.jsonl file. Default: current directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the last N evaluation rows.",
    )
    parser.add_argument(
        "--description-width",
        type=int,
        default=72,
        help="Maximum width for the description column.",
    )
    args = parser.parse_args()

    run_dir, log_path = _resolve_run_and_log(args.path)
    rows = _load_rows(log_path)
    if args.limit is not None and args.limit >= 0:
        rows = rows[-args.limit :]

    print(f"run: {run_dir}")
    print(f"log: {log_path}")
    print(f"rows: {len(rows)}")
    if not rows:
        return 0
    print()

    header = f"{'iter':>4}  {'score':>18}  {'status':<8}  description"
    print(header)
    print(f"{'-' * 4}  {'-' * 18}  {'-' * 8}  {'-' * min(args.description_width, 11)}")
    for row in rows:
        iteration = row.get("iteration", "")
        result = row.get("result", {}) or {}
        score = _format_score(result.get("score"))
        status = str(result.get("status", ""))
        description = _truncate(str(row.get("description", "")), args.description_width)
        print(f"{str(iteration):>4}  {score:>18}  {status:<8}  {description}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
