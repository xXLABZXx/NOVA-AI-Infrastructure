# SECURITY.md

## Threat model

NOVA AI Infrastructure ingests **untrusted content from the public internet**
and turns it into training data. The primary threats we defend against:

1. **SSRF / network pivoting** — a malicious URL tricking the fetcher into
   hitting internal services (cloud metadata, localhost, private ranges).
2. **Malicious downloads** — malware or oversized files aimed at the training host.
3. **Prompt injection** — crawled pages containing instructions that attempt to
   override system behavior, exfiltrate secrets, or trigger privileged actions.
4. **Secret leakage** — API keys or tokens committed to git or written to logs.
5. **Data poisoning / leakage** — eval contamination (test data leaking into
   training), silent dataset mutation, unverified teacher outputs.
6. **Supply chain** — compromised or vulnerable dependencies.

## Controls

- **Sandboxed fetching** (`src/nova/sandbox/`): domain allowlist/blocklist,
  DNS-resolution-based SSRF checks rejecting private/loopback/link-local IPs,
  explicit blocks for cloud metadata endpoints (e.g. `169.254.169.254`),
  timeouts, retries with backoff, token-bucket rate limiting, per-download size
  caps, content-type validation by magic bytes, SHA-256 hashing, and quarantine
  isolation. Network access is **disabled by default**.
- **Container isolation**: the sandbox Docker image runs as non-root with a
  read-only root filesystem, dropped capabilities, and only `quarantine/` and
  `logs/` mounted writable. Training/data dirs are never mounted.
- **Untrusted content handling** (`src/nova/security/injection_guard.py`):
  crawled text is wrapped in delimiters and scanned for instruction-like
  patterns before downstream use; it is always treated as data, never as
  instructions.
- **Safe deserialization** (`src/nova/security/safe_json.py`): strict JSON
  parsing with size limits; no `pickle`/`eval` anywhere in the codebase.
- **Secret hygiene**: credentials only via environment variables / GitHub
  Secrets; `.gitignore`d `.env`; gitleaks in CI; log redaction in
  `nova.monitoring` (keys/tokens replaced with `[REDACTED]`); secret scanning
  helper in `nova.security.secret_scan`.
- **Dataset integrity**: raw store is append-only; every modification proposal
  gets a changelog entry (record id, change, reason, validation, approval);
  dataset releases carry `manifest.json` with SHA-256s.
- **Split discipline**: `train/val/test/challenge` splits are enforced by
  `nova.training.datasets`; held-out splits can never enter the training loader.
- **Dependencies**: pinned in `pyproject.toml`; `pip-audit` in CI.

## Reporting

This is a private project. Report suspected vulnerabilities directly to the
repository owner — do not open public issues with exploit details. Include:
what you found, how to reproduce, and the potential impact.
