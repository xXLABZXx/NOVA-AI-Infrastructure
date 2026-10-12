#!/usr/bin/env bash
# Run the full test suite.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest -q "$@"
