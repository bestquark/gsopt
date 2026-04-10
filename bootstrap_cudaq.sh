#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv sync --python 3.12
source .venv/bin/activate
python -m ensurepip >/dev/null 2>&1 || true
python -m pip install cudaq
