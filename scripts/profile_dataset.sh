#!/usr/bin/env bash
# Profile the base dataset: streams all parts, reports template collapse.
# Usage: ./scripts/profile_dataset.sh [parts-glob] [report.json]
set -euo pipefail
cd "$(dirname "$0")/.."
PARTS="${1:-${NOVA_BASE_PARTS:-NOVA_DATASET_#0007_part-*.txt}}"
OUT="${2:-data/factory/profile-report.json}"
python3 -m nova.datasets.profile --parts "$PARTS" --out "$OUT"
