#!/usr/bin/env bash
# Example: crawl seed URLs into data/raw via the sandbox.
# Usage: ./scripts/crawl.sh https://en.wikipedia.org/wiki/Artificial_intelligence
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m nova.crawlers.runner --config configs/config.yaml --out data/raw "$@"
