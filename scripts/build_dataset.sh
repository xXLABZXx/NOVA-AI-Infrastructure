#!/usr/bin/env bash
# Run the full dataset factory: profile -> diversify -> ingest -> mix.
# Usage: ./scripts/build_dataset.sh [--config configs/config.yaml] [--workdir data/factory]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="configs/config.yaml"
WORKDIR="data/factory"
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
python3 -m nova.datasets.factory --config "$CONFIG" --workdir "$WORKDIR"
