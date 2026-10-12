# Dataset factory guide

The factory continues the base dataset (`NOVA_DATASET_#0007`, 20 parts,
17.4M records) into dataset **v8+**: it keeps your prompts and ids, fixes the
answer-template collapse, and mixes in genuinely new data. Everything streams —
the 8.9GB base is never loaded into RAM.

## The problem it solves

Profiling the base shows the collapse: only ~52 unique answers across 17.4M
records (e.g. 3M `general_chat` records share 2 answers). Training on that
teaches the model to parrot templates. The factory:

1. **Profiles** — measures collapse, diversity, length stats (`profile.py`).
2. **Diversifies** — rewrites collapsed answers with a teacher model
   (DeepSeek), preserving prompt + id, stamping provenance (`diversify.py`).
3. **Ingests** — pulls bulk web text from Common Crawl (free HTTP, no AWS key)
   for configured domains (`commoncrawl.py`).
4. **Mixes** — exact-dedups new data against the base, near-dedups among new
   records, balances categories, assigns leak-free splits, writes versioned
   output + manifest (`mix.py`).

## Quick start

```bash
# 1. Profile the base (works on the zip directly)
./scripts/profile_dataset.sh \
  "$HOME/workspace/nova-dataset/dataset007.zip:NOVA_DATASET_#0007_part-*.txt"

# 2. Dry-run the diversification plan (no API key needed)
python3 -m nova.datasets.diversify \
  --parts "$HOME/workspace/nova-dataset/dataset007.zip:NOVA_DATASET_#0007_part-*.txt" \
  --out /tmp/diversified.jsonl --dry-run --limit 1000

# 3. Full factory run (stages toggled in configs/config.yaml)
./scripts/build_dataset.sh
```

## Config reference (`datasets:` section)

| Key | Meaning |
|---|---|
| `enabled` | master switch |
| `base_parts_glob` | glob for base parts; `archive.zip` or `archive.zip:member-glob` also works |
| `version` | output version, e.g. `"8"` → `dataset-v8-*.jsonl` |
| `stages.{profile,diversify,ingest,mix}` | run each stage independently |
| `diversify.min_template_count` | answers seen ≥ this often count as collapsed (default 1000) |
| `diversify.limit` | max rewrites per run (0 = unlimited) |
| `diversify.dry_run` | plan + cost estimate, no API calls |
| `diversify.resume_from` | record id to resume after |
| `diversify.teacher.*` | provider, model, `max_calls_per_run`, `temperature` (0.9 default: we want diversity) |
| `commoncrawl.crawl_id` | e.g. `CC-MAIN-2026-39`; empty = latest |
| `commoncrawl.domains` | domains to pull, e.g. `en.wikipedia.org` |
| `commoncrawl.max_records` | cap per run |
| `mixer.base_hash_db` | SQLite fingerprint store (~1–2GB for 17.4M; built once) |
| `mixer.category_weights` | target mix ratios, normalized |
| `mixer.split_ratios` | `{train: 80, val: 10, test: 5, challenge: 5}` |
| `mixer.enable_near_dedup` | SimHash near-dedup among *new* records |
| `mixer.extra_sources` | extra inputs, e.g. crawler JSONL |

## Teacher cost notes (diversify)

- Cost estimate is printed by every dry-run:
  `est_cost_usd` from `cost_per_mtok_in/out` (DeepSeek chat defaults).
- Rewriting is the expensive step: each collapsed record = 1 API call.
  Start with `--limit 1000`, inspect `diversified.jsonl`, then scale.
- Every rewrite is validated (non-empty, ≠ template, sane length, no prompt
  echo) and stamped with `provenance: {rewritten_by, rewrite_model,
  original_answer_sha256, rewrite_style}`. Rejected outputs are logged with
  reasons, never silently kept.
- Requires `DEEPSEEK_API_KEY` (or `TEACHER_API_KEY`) in the environment.
  Without it the stage degrades to a dry-run plan automatically.

## Common Crawl notes

- Endpoints used: `index.commoncrawl.org` (index API) and
  `data.commoncrawl.org` (WET byte ranges). Both are free HTTP; no AWS
  credentials.
- The ingester is polite: 0.5s between range fetches, retries with backoff,
  and a checkpoint file so interrupted runs resume without re-downloading.
- Output records are `kind: "text"` with `provenance: {source: commoncrawl,
  crawl, warc_date}` and the source URL.

## Mixer details

- **Exact dedup vs base**: SHA-256 of normalized content, looked up in the
  SQLite store. New records already in the 17.4M base are dropped with reason
  `exact_dup_of_base`.
- **Near dedup**: SimHash (Hamming ≤ threshold) among new records only.
- **Balancing**: deterministic hash subsampling per category toward
  `category_weights`; over-quota drops get `category_over_quota`.
- **Splits**: `SplitEnforcer` assigns by id hash — deterministic, and
  `load_for_training` rejects any held-out leakage.
- **Outputs**: `dataset-v{N}-{split}-part-*.jsonl`, `manifest.json`
  (file SHAs, counts, config snapshot), `rejected.jsonl` (every drop has an
  id + reason code).

## Scaling honesty

- The SQLite base store build is one-time but slow (~17.4M inserts; tens of
  minutes) and ~1–2GB on disk.
- SimHash near-dedup is O(n²) in memory — fine for hundreds of thousands of
  *new* records, not for re-processing the full 17.4M. Near-dedup runs only
  over new intake, never the base.
- Teacher rewriting is the cost bottleneck; the dry-run estimate exists so
  you can budget before spending.
