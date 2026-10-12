# Changelog

All notable changes to NOVA AI Infrastructure are recorded here.

## [0.1.0] - 2026-10-12
### Added
- Initial scaffold: `src/nova` package with sandbox, crawlers, ingestion,
  deduplication, curation, teacher, training, evaluation, monitoring, security.
- Central YAML configuration with environment-variable overrides (`configs/config.yaml`).
- Controlled HTTP sandbox with allowlist/blocklist, SSRF protection, rate
  limiting, download caps, quarantine, and JSONL audit logging.
- Modular crawlers (general, docs, code, papers) with robots.txt compliance,
  sitemap/RSS discovery, URL dedup, incremental fetching.
- Append-only raw document store; exact (SHA-256) and near (SimHash) dedup.
- Curation pipeline raw -> cleaned -> approved/rejected with per-record
  changelog and dataset manifests.
- Optional DeepSeek/OpenAI-compatible teacher integration (disabled without API key).
- PyTorch training integration (lazy torch import) with checkpoint compatibility
  checks and train/val/test/challenge split enforcement.
- Evaluation harness with exact-match/accuracy metrics.
- Monitoring (redacted JSONL logging, health checks, quotas) and security
  helpers (input validation, safe deserialization, secret scanning,
  prompt-injection guard).
- Pytest suite with synthetic fixtures; CI workflow (tests, gitleaks, pip-audit).
- Dockerfile + docker-compose for the hardened sandbox service.
