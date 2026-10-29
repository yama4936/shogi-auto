#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'CHECK'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
print("Python:", sys.version.split()[0])
CHECK
if [[ ! -x .venv/bin/python ]]; then
    python -m venv .venv
fi
.venv/bin/python -m pip install --no-cache-dir -e .
.venv/bin/python -m unittest discover -s tests -v
echo "Ready: source .venv/bin/activate"
echo "Play: .venv/bin/python -m shogi_auto.web"
