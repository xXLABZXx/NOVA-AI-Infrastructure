# Operations

## Enabling / disabling modules

Edit `configs/config.yaml`:

```yaml
modules:
  teacher: false   # keep disabled until DEEPSEEK_API_KEY is configured
  crawlers: true
```

Or override per-environment without editing the file:

```bash
NOVA_MODULES_TEACHER=false ./scripts/curate.sh
NOVA_SANDBOX_REQUESTS_PER_MINUTE=10 ./scripts/crawl.sh <url>
```

List form: `NOVA_SANDBOX_ALLOWLIST="en.wikipedia.org,arxiv.org"`.

## Daily workflow

```bash
./scripts/run_tests.sh          # full test suite
./scripts/crawl.sh <seed-url>   # collect -> data/raw/
./scripts/curate.sh             # curate  -> data/{cleaned,approved,rejected}/
./scripts/train.sh data/approved/dataset-v1.jsonl   # training preflight
python -m nova.evaluation.harness --predictions out.jsonl --expected eval.jsonl
```

## Sandbox service (Docker)

```bash
cp .env.example .env   # fill in keys; never commit .env
docker compose up --build sandbox
```

The container runs as non-root, read-only root FS, no host network, with only
`quarantine/` and `logs/` mounted writable. Promote downloads from `quarantine/`
to `data/raw/` only after validation.

## Monitoring

- All components emit JSONL events to `logs/` via `nova.monitoring` (secrets redacted).
- `python -m nova.monitoring.health` prints a health summary: module status,
  disk usage vs quota, quarantine size, recent errors.
- Audit trail for sandbox fetches: `logs/sandbox-audit.jsonl`
  (URL, timestamp, status, bytes, sha256).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SandboxDisabledError` | `sandbox.enabled: false` | enable in config or `NOVA_SANDBOX_ENABLED=true` |
| `DomainBlockedError` | host not in allowlist | add to `sandbox.allowlist` |
| `SSRFBlockedError` | private/loopback IP or metadata endpoint | expected — do not bypass |
| Crawl yields nothing | robots.txt disallow / rate limit | check `logs/crawler-failures.jsonl` |
| Teacher returns "disabled" | no API key | set `DEEPSEEK_API_KEY` and `modules.teacher: true` |
| Checkpoint rejected | arch/vocab/tokenizer mismatch | see error detail; do not force-load |
| Tests fail on import | package not installed | `pip install -e ".[dev]"` |
