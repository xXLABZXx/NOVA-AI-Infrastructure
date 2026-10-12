# NOVA AI Infrastructure

Secure, modular infrastructure for the **NOVA** AI model: controlled internet
access, web crawling and research, dataset curation and versioning, optional
teacher-model integration (DeepSeek), PyTorch training integration, and
evaluation — with security as a core requirement, not an afterthought.

> **Status:** v0.1.0 scaffold with working core components. See
> [docs/roadmap.md](docs/roadmap.md) for what's next.

## Layout

```
NOVA-AI-Infrastructure/
├── src/nova/            # Python packages
│   ├── config.py        # central YAML config + env overrides
│   ├── sandbox/         # controlled HTTP client (the ONLY network-touching code)
│   ├── crawlers/        # modular crawlers (general, docs, code, papers)
│   ├── ingestion/       # append-only raw store, cleaning, normalization
│   ├── deduplication/   # exact (SHA-256) + near-duplicate (SimHash)
│   ├── curation/        # raw → cleaned → approved/rejected + changelog + manifests
│   ├── teacher/         # optional DeepSeek/OpenAI-compatible reviewer
│   ├── training/        # PyTorch integration, checkpoints, split enforcement
│   ├── evaluation/      # eval harness, metrics, reports
│   ├── monitoring/      # redacted JSONL logging, health checks, quotas
│   └── security/        # validation, safe JSON, secret scan, injection guard
├── configs/config.yaml  # module toggles, sandbox limits, crawl & training settings
├── tests/               # pytest suite with synthetic fixtures (no network, no torch)
├── scripts/             # run_tests.sh, crawl.sh, curate.sh, train.sh
├── docs/                # architecture, operations, colab, roadmap
├── data/                # raw/cleaned/approved/rejected (manifests only in git)
├── .github/workflows/ci.yml
├── Dockerfile / docker-compose.yml   # hardened sandbox service
└── README.md SECURITY.md CHANGELOG.md .env.example pyproject.toml
```

See [docs/architecture.md](docs/architecture.md) for the data flow and trust boundaries.

## Installation

Requirements: Python 3.10+, pip. Docker optional (sandbox service). A GPU host or
Google Colab for actual training.

```bash
git clone <this-repo> && cd NOVA-AI-Infrastructure
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # dev/test deps
# pip install -e ".[torch]"      # only on training hosts (Colab/GPU)
cp .env.example .env             # fill in keys as needed; never commit .env
./scripts/run_tests.sh
```

## Configuration

Everything lives in `configs/config.yaml`. Enable/disable modules independently:

```yaml
modules:
  sandbox: true
  teacher: false     # needs DEEPSEEK_API_KEY
```

Environment variables override any key (`NOVA_<SECTION>_<KEY>`):

```bash
NOVA_MODULES_TEACHER=false
NOVA_SANDBOX_REQUESTS_PER_MINUTE=10
NOVA_SANDBOX_ALLOWLIST="en.wikipedia.org,arxiv.org"
DEEPSEEK_API_KEY=sk-...   # teacher credentials: env only, never in files
```

## Usage

```bash
./scripts/run_tests.sh                       # pytest, hermetic (no network/torch)
./scripts/crawl.sh https://en.wikipedia.org/wiki/Artificial_intelligence
./scripts/curate.sh                          # raw → cleaned → approved/rejected
./scripts/train.sh data/approved/dataset-v1.jsonl   # training preflight checks
python -m nova.evaluation.harness --predictions out.jsonl --expected eval.jsonl
python -m nova.monitoring.health             # health + quota summary
```

Direct module entry points (all support `--help`):

- `python -m nova.sandbox.worker --url <url>` — fetch one URL through the sandbox
- `python -m nova.crawlers.runner --config configs/config.yaml --out data/raw <seeds...>`
- `python -m nova.curation.pipeline --config configs/config.yaml --raw data/raw ...`
- `python -m nova.training.preflight --config configs/config.yaml <dataset.jsonl>`

## Docker sandbox

```bash
cp .env.example .env   # fill secrets; never commit
docker compose up --build sandbox
```

Hardened: non-root user, read-only root FS, dropped capabilities, no host
network, only `quarantine/` + `logs/` mounted writable. Promote downloads from
`quarantine/` to `data/raw/` only after validation.

## Google Colab

See [docs/colab.md](docs/colab.md): prepare + verify an approved dataset locally,
mount it in Colab, run `nova.training.preflight`, then train with your existing
NOVA loop using `TrainingConfig`, `CheckpointManager`, and `SplitEnforcer`.

## Security

Read [SECURITY.md](SECURITY.md). Highlights:

- Network access is **disabled by default**; the sandbox enforces
  allowlist/blocklist, SSRF protection (private IPs, loopback, cloud metadata
  endpoints blocked), rate limits, download caps, and quarantine isolation.
- Crawled content is **untrusted data** — delimited, scanned for
  instruction-like patterns, never treated as instructions.
- **Never commit** API keys, tokens, datasets, or checkpoints. Use `.env`
  (gitignored) locally and GitHub Secrets in CI. CI runs gitleaks + pip-audit.
- Raw data is append-only; every dataset modification has a changelog entry;
  releases carry SHA-256 manifests. Val/test/challenge splits can never leak
  into training.

## Troubleshooting

See [docs/operations.md](docs/operations.md) for the troubleshooting table.

## Publishing to GitHub (private repo)

The GitHub App used for automation may lack repository-creation permission; if
so, publish manually — it takes a minute:

```bash
cd NOVA-AI-Infrastructure
git init -b main
git add .
git commit -m "Initial commit: NOVA AI Infrastructure v0.1.0"
# Create an empty PRIVATE repository on github.com first (do not initialize with README),
# then:
git remote add origin git@github.com:<your-username>/NOVA-AI-Infrastructure.git
git push -u origin main
```

Or with the GitHub CLI:

```bash
gh repo create NOVA-AI-Infrastructure --private --source=. --push
```

After pushing, add secrets at **Settings → Secrets and variables → Actions**:
`DEEPSEEK_API_KEY` (only if you use the teacher). Never commit `.env`.
