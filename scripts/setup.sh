#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'CHECK'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
print("Python:", sys.version.split()[0])
CHECK
python -m unittest discover -s tests -v
