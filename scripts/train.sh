#!/usr/bin/env bash
# Example: preflight checks for a training run (works without torch installed).
# Full training runs on GPU hosts / Colab - see docs/colab.md.
# Usage: ./scripts/train.sh data/approved/dataset-v1.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m nova.training.preflight --config configs/config.yaml "$@"
