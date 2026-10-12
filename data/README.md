# Data directories

This repository intentionally contains **no real datasets, checkpoints, or model weights**.
Only this README and `.gitkeep` files are committed.

| Directory | Purpose | Written by |
|---|---|---|
| `raw/` | Append-only raw crawled/collected documents (JSONL). Never modified in place. | crawlers, ingestion |
| `cleaned/` | Normalized, deduplicated documents awaiting review. | curation pipeline |
| `approved/` | Human- or policy-approved training examples (JSONL). | curation (with approval) |
| `rejected/` | Documents rejected by quality/reliability gates, kept for audit. | curation pipeline |

Every document carries provenance (`source URL`, `retrieved_at`, `sha256`, license note).
Dataset releases are versioned via `manifest.json` files (see `src/nova/curation/versions.py`).
Large or private datasets must live outside git (object storage / DVC remote); only
manifests and hashes belong in the repository.
