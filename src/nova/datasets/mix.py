"""Dataset mixer: dedup new data against the base, balance, split, version.

Pipeline (all streaming, disk-backed — never holds the dataset in RAM):
  1. build/refresh a SQLite SHA-256 store of base fingerprints (one-time,
     ~1-2GB for 17.4M records);
  2. stream new sources (diversified JSONL, Common Crawl text, crawler JSONL):
     exact-dedup vs the base store, near-dedup among new records (SimHash);
  3. category balancing via deterministic hash subsampling to target weights;
  4. split assignment with ``SplitEnforcer`` (train/val/test/challenge,
     hash-deterministic on id, leak-checked);
  5. write ``dataset-v{N}`` split parts + ``manifest.json``; every dropped
     record lands in ``rejected.jsonl`` with a reason code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from nova.datasets.base import PartStats, fingerprint, iter_jsonl_parts, write_jsonl


# ---------------------------------------------------------------------------
# SQLite base hash store
# ---------------------------------------------------------------------------

def build_base_hash_store(base_pattern: str, db_path: str,
                          *, batch: int = 20000) -> dict:
    """Stream base parts into a SQLite fingerprint store. Returns stats."""
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS base_hashes (h TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    stats = PartStats()
    n = 0
    buf: list[tuple[str]] = []
    t0 = time.time()
    for rec in iter_jsonl_parts(base_pattern, stats=stats):
        buf.append((fingerprint(rec),))
        n += 1
        if len(buf) >= batch:
            db.executemany("INSERT OR IGNORE INTO base_hashes (h) VALUES (?)", buf)
            db.commit()
            buf.clear()
    if buf:
        db.executemany("INSERT OR IGNORE INTO base_hashes (h) VALUES (?)", buf)
        db.commit()
    db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('records', ?)",
               (str(n),))
    db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('built_at', ?)",
               (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
    db.commit()
    size = Path(db_path).stat().st_size
    db.close()
    return {"base_records": n, "malformed": stats.malformed,
            "db_path": db_path, "db_bytes": size,
            "seconds": round(time.time() - t0, 1)}


class BaseHashStore:
    """Read-only lookup against the base fingerprint DB."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA query_only=ON")

    def contains(self, fp: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM base_hashes WHERE h = ? LIMIT 1", (fp,)).fetchone()
        return row is not None

    def close(self):
        self.db.close()


# ---------------------------------------------------------------------------
# Source normalization
# ---------------------------------------------------------------------------

def normalize_source_record(rec: dict, source_name: str) -> dict:
    """Map heterogeneous inputs to the canonical record dict."""
    d = dict(rec)
    d.setdefault("id", f"{source_name}-{hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]}")
    # crawler JSONL uses url/text/title
    if "text" in d and "prompt" not in d:
        d["kind"] = "text"
    d.setdefault("kind", "sft")
    d.setdefault("category", "web" if d.get("kind") == "text" else "general")
    d.setdefault("source", source_name)
    d.setdefault("synthetic", False)
    d.setdefault("provenance", {})
    return d


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------

REJECT_REASONS = (
    "exact_dup_of_base",
    "near_dup_of_new",
    "category_over_quota",
    "empty_content",
)


def _content_empty(rec: dict) -> bool:
    if rec.get("kind") == "text":
        return not (rec.get("text") or "").strip()
    return not (rec.get("prompt") or "").strip() or not (rec.get("answer") or "").strip()


def mix(
    sources: list[dict],
    base_db: str,
    out_dir: str,
    *,
    version: str = "8",
    category_weights: dict | None = None,
    split_ratios: dict | None = None,
    near_dup_threshold_bits: int = 8,
    enable_near_dedup: bool = True,
    records_per_part: int = 500_000,
    seed: int = 42,
    min_text_chars: int = 50,
) -> dict:
    """Run the full mix. Returns stats.

    ``sources``: [{"name": ..., "pattern": ...}, ...] (JSONL globs).
    """
    from nova.deduplication.near import NearDeduper
    from nova.training.datasets import SplitEnforcer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_accepted = out_dir / "_accepted.jsonl"
    rejected_path = out_dir / "rejected.jsonl"
    if tmp_accepted.exists():
        tmp_accepted.unlink()

    store = BaseHashStore(base_db)
    near = NearDeduper(threshold_bits=near_dup_threshold_bits) if enable_near_dedup else None
    counts = Counter()
    rejected = Counter()

    def reject(rec_id: str, reason: str, src: str):
        rejected[reason] += 1
        with open(rejected_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": rec_id, "reason": reason,
                                 "source": src}) + "\n")

    # -- pass 1: dedup -> accepted temp file -------------------------------
    accepted_n = 0
    with open(tmp_accepted, "w", encoding="utf-8") as out:
        for src in sources:
            sname = src["name"]
            for rec in iter_jsonl_parts(src["pattern"]):
                rec = normalize_source_record(rec, sname)
                rid = str(rec.get("id", ""))
                if _content_empty(rec) or len(
                        rec.get("text", "") or rec.get("answer", "")) < min_text_chars:
                    reject(rid, "empty_content", sname)
                    continue
                fp = fingerprint(rec)
                if store.contains(fp):
                    reject(rid, "exact_dup_of_base", sname)
                    continue
                text_for_near = rec.get("text") or (
                    f"{rec.get('prompt', '')} {rec.get('answer', '')}")
                if near is not None and near.is_near_duplicate(text_for_near):
                    reject(rid, "near_dup_of_new", sname)
                    continue
                if near is not None:
                    near.add(text_for_near)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[rec.get("category", "general")] += 1
                accepted_n += 1
    store.close()

    # -- pass 2: balancing quotas -------------------------------------------
    total = sum(counts.values())
    weights = dict(category_weights or {})
    if weights and total:
        wsum = sum(weights.get(c, 1.0) for c in counts)
        keep_pct = {}
        for cat, cnt in counts.items():
            target = total * (weights.get(cat, 1.0) / wsum)
            keep_pct[cat] = min(100.0, (target / cnt * 100) if cnt else 0.0)
    else:
        keep_pct = {c: 100.0 for c in counts}

    # -- pass 3: split + write ----------------------------------------------
    enforcer = SplitEnforcer(ratios=split_ratios) if split_ratios else SplitEnforcer()
    writers: dict[str, dict] = {}  # split -> {"fh":..., "part":int, "n":int}
    split_counts: Counter = Counter()
    balanced_out = Counter()

    def writer_for(split: str):
        st = writers.get(split)
        if st is None or st["n"] >= records_per_part:
            if st:
                st["fh"].close()
            part = (st["part"] + 1) if st else 0
            path = out_dir / f"dataset-v{version}-{split}-part-{part:03d}.jsonl"
            st = {"fh": open(path, "w", encoding="utf-8"),
                  "part": part, "n": 0}
            writers[split] = st
        return st

    with open(tmp_accepted, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cat = rec.get("category", "general")
            # deterministic hash subsampling for balancing
            bucket = int(hashlib.sha256(
                f"{seed}:{rec.get('id', '')}".encode()).hexdigest(), 16) % 10000
            if bucket >= keep_pct.get(cat, 100.0) * 100:
                reject(str(rec.get("id", "")), "category_over_quota",
                       rec.get("source", ""))
                continue
            split = enforcer.assign(str(rec.get("id", "")))
            rec["split"] = split
            st = writer_for(split)
            st["fh"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["n"] += 1
            split_counts[split] += 1
            balanced_out[cat] += 1
    for st in writers.values():
        st["fh"].close()
    tmp_accepted.unlink(missing_ok=True)

    # -- manifest ------------------------------------------------------------
    from nova.curation.versions import write_manifest
    manifest = write_manifest(
        out_dir, version,
        extra={
            "kind": "nova-dataset",
            "base": "NOVA_DATASET_#0007",
            "splits": dict(split_counts),
            "categories_in": dict(counts),
            "categories_out": dict(balanced_out),
            "keep_pct": keep_pct,
            "rejected": dict(rejected),
            "split_ratios": enforcer.ratios,
            "seed": seed,
            "sources": [s["name"] for s in sources],
        },
    )
    return {
        "accepted": accepted_n,
        "balanced_out": sum(balanced_out.values()),
        "splits": dict(split_counts),
        "rejected": dict(rejected),
        "manifest": str(manifest),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset factory mixer.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--build-hashes", action="store_true",
                        help="Only (re)build the base hash store, then exit")
    args = parser.parse_args(argv)

    from nova.config import load_config
    cfg = load_config(args.config)
    ds = cfg.section("datasets") if hasattr(cfg, "section") else cfg.get("datasets", {})
    mx = dict(ds.get("mixer", {}))

    base_pattern = ds.get("base_parts_glob", "")
    db_path = mx.get("base_hash_db", "data/base_hashes.sqlite")
    if args.build_hashes or not Path(db_path).exists():
        print("building base hash store (one-time, streaming)...")
        print(json.dumps(build_base_hash_store(base_pattern, db_path), indent=2))
        if args.build_hashes:
            return 0

    stats = mix(
        sources=mx.get("sources", []),
        base_db=db_path,
        out_dir=mx.get("output_dir", "data/dataset-v8"),
        version=str(ds.get("version", "8")),
        category_weights=mx.get("category_weights"),
        split_ratios=mx.get("split_ratios"),
        near_dup_threshold_bits=int(mx.get("near_dup_threshold_bits", 8)),
        enable_near_dedup=bool(mx.get("enable_near_dedup", True)),
        records_per_part=int(mx.get("records_per_part", 500_000)),
        seed=int(mx.get("seed", 42)),
        min_text_chars=int(mx.get("min_text_chars", 50)),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
