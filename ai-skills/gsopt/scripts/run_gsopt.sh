#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: run_gsopt.sh <command...>" >&2
  exit 1
fi

if [ "$1" = "--" ]; then
  shift
fi

run_dir="${GSOPT_RUN_DIR:-$PWD}"
log_root="${run_dir}/logs/launcher"
stamp="$(date -u +%Y%m%d_%H%M%S)"
stdout_file="${log_root}/${stamp}.stdout.log"
stderr_file="${log_root}/${stamp}.stderr.log"
heartbeat_file="${log_root}/heartbeat.json"
pid_file="${log_root}/worker.pid"

mkdir -p "$log_root"

bash "$(dirname "$0")/heartbeat.sh" "$heartbeat_file" 20 &
HB_PID=$!

cleanup() {
  kill "$HB_PID" >/dev/null 2>&1 || true
  if [ -n "${WD_PID:-}" ]; then
    kill "$WD_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$@" >>"$stdout_file" 2>>"$stderr_file" &
WORKER_PID=$!
echo "$WORKER_PID" >"$pid_file"

python3 "$(dirname "$0")/watchdog.py" "$heartbeat_file" "$pid_file" 180 15 &
WD_PID=$!

wait "$WORKER_PID"
