#!/usr/bin/env bash
# Example: run the curation pipeline raw -> cleaned -> approved/rejected.
# Usage: ./scripts/curate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m nova.curation.pipeline --config configs/config.yaml \
  --raw data/raw --cleaned data/cleaned \
  --approved data/approved --rejected data/rejected
