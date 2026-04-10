#!/usr/bin/env bash
set -euo pipefail

hb_file="${1:-${GSOPT_RUN_DIR:-$PWD}/logs/launcher/heartbeat.json}"
interval="${2:-20}"

mkdir -p "$(dirname "$hb_file")"

while true; do
  python3 - <<'PY' "$hb_file"
import json
import os
import sys
import time

path = sys.argv[1]
payload = {
    "ts": time.time(),
    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pid": os.getpid(),
    "status": "alive",
}
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(payload, handle, indent=2)
os.replace(tmp, path)
PY
  sleep "$interval"
done
