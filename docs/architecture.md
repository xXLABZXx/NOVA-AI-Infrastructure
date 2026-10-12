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
├── crawlers/        concurrent crawlers (general/docs/code/papers): persistent
│                    SQLite frontier, per-host politeness scheduler, article
│                    extraction, sitemap/feed discovery — all fetches go through
│                    the sandbox
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

## Web crawler subsystem

The crawlers (`general` / `docs` / `code` / `papers`, all extending
`BaseCrawler`) are built around three pieces:

- **Frontier** (`crawlers/frontier.py`): a SQLite work queue storing every URL
  with priority, depth, retry state, `next_fetch_at`, sitemap level and a
  `revalidate` flag. Priorities favour shallow pages, earlier seeds and
  fresh feed items. The same database hosts the robots.txt cache (with TTL)
  and per-host state (page budgets, backoff windows), so an interrupted crawl
  resumes without refetching completed pages — `runner.py` reuses
  `<out>/.frontier.db` and appends to `crawl.jsonl` on resume.
- **Scheduler** (`crawlers/scheduler.py`): a `ThreadPoolExecutor` dispatcher
  (bounded `concurrency`, default 10) with per-host politeness. Each host gets
  a minimum inter-request delay — robots.txt `crawl-delay` when declared,
  otherwise `per_host_delay_seconds` — plus an optional global token bucket
  (`requests_per_minute`). 429/5xx responses trigger host-level exponential
  backoff with jitter; per-host (`max_pages_per_host`) and global
  (`max_pages_total`) budgets bound the crawl. All fetching happens in worker
  threads through the sandbox HTTP client, so SSRF protection, domain
  allow/blocklists and download caps stay on the request path.
- **Extraction** (`crawlers/extract.py`): readability-style main-content
  extraction — candidate blocks are scored by text density (length minus
  link-text penalty), boilerplate (`nav`/`footer`/ads/chrome by tag and
  class patterns) is stripped, and title (og:title → `<title>`), meta
  description, publish date, author, language + confidence are captured.
  Every record carries a `quality` dict (word count, text-to-HTML ratio,
  boilerplate ratio, aggregate 0..1 score).

Supporting pieces: `canonical.py` (tracking-param stripping, query sorting,
fragment/default-port removal — dedup happens *before* fetching),
`discovery.py` (recursive sitemap indexes with `<lastmod>`-aware incremental
decisions, RSS/Atom parsing with pub dates, freshest first), `robots.py`
(TTL-cached robots.txt with crawl-delay and `Sitemap:` collection),
`store.py` (thread-safe seen ledger with ETag/Last-Modified revalidation —
a 304 records `not_modified` without re-extracting). Permanently failed URLs
go to a dead-letter log (`dead_letter.jsonl`) with reason codes and per-URL
attempt history; one bad URL or host can never stall the crawl.

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
