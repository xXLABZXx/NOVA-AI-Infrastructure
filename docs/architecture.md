# Architecture

NOVA AI Infrastructure is a set of independent Python packages under `src/nova/`.
Each package is toggled via `configs/config.yaml` (`modules.<name>`) and can be
used standalone — constructors take explicit keyword arguments, with an optional
`from_config(section_dict)` classmethod for wiring from the central config.

```
configs/config.yaml
        │
        ▼
src/nova/
├── config.py        YAML load + NOVA_ env overrides + validation
├── security/        input validation, safe JSON, secret scan, injection guard, audit
├── monitoring/      redacted JSONL logging, health checks, disk quotas
├── sandbox/         controlled HTTP client (allowlist/blocklist, SSRF guard,
│                    rate limit, download caps, quarantine, audit log)
├── crawlers/        modular crawlers (general/docs/code/papers) built on the sandbox
├── ingestion/       append-only raw JSONL store, normalization, boilerplate removal
├── deduplication/   exact (SHA-256) + near-duplicate (SimHash) detection
├── curation/        raw → cleaned → approved/rejected, scoring, manifests, changelog
├── teacher/         optional DeepSeek/OpenAI-compatible reviewer (disabled w/o key)
├── training/        PyTorch integration: checkpoints, splits, loss tracking
└── evaluation/      eval harness + metrics + reports
```

## Data flow

1. **Collect**: crawlers fetch public pages through the sandbox → raw JSONL in `data/raw/`
   (append-only; originals are never modified in place).
2. **Ingest/dedupe**: normalization, boilerplate removal, language filter,
   exact + near-duplicate detection.
3. **Curate**: quality + source-reliability scoring → `data/cleaned/` →
   human/policy approval → `data/approved/` (or `data/rejected/` for audit).
   Every proposed modification is recorded in a per-record changelog
   (`record_id`, proposed change, reason, validation, approval status).
4. **Version**: dataset releases get `manifest.json` (file SHA-256s, version id,
   timestamp) for reproducibility.
5. **Train** (Colab/GPU host): split-enforced loading (`train/val/test/challenge`),
   checkpoint compatibility checks, loss tracking, experiment snapshots.
6. **Evaluate**: harness runs model outputs against JSONL eval sets; challenge
   split is reserved for difficult independent evaluation.

## Trust boundaries

- Crawled web content is **untrusted data**. It is wrapped/delimited and scanned
  by `security/injection_guard.py` before it can influence any downstream
  decision; it can never override system instructions or trigger privileged actions.
- The sandbox is the **only** component allowed to touch the network, and only
  when `modules.sandbox` / `sandbox.enabled` is true. Downloads land in
  `quarantine/`, isolated from `data/` and training, until validated.
- Secrets live in environment variables / GitHub Secrets only. They are never
  logged (monitoring redacts them) and never committed (`.gitignore` + gitleaks in CI).
- The teacher model **proposes**; it never overwrites approved data or checkpoints.
