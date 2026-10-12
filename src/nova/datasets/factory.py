"""Dataset factory orchestrator.

``python -m nova.datasets.factory --config configs/config.yaml``

Runs the enabled stages from the ``datasets:`` config section, in order:
profile -> diversify -> ingest (Common Crawl) -> mix. Each stage is
independently enableable; stages communicate through files on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load_datasets_config(config_path: str) -> tuple[dict, object]:
    from nova.config import load_config
    cfg = load_config(config_path)
    ds = cfg.section("datasets") if hasattr(cfg, "section") else cfg.get("datasets", {})
    return dict(ds or {}), cfg


def run_profile(ds: dict, out_dir: Path) -> dict:
    from nova.datasets.profile import profile_parts
    print("[factory] stage: profile")
    report = profile_parts(ds["base_parts_glob"])
    (out_dir / "profile-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"  records={report['total_records']} "
          f"collapse_ratio={report['answer_collapse_ratio']}")
    return report


def run_diversify(ds: dict, out_dir: Path) -> dict:
    from nova.datasets.diversify import diversify
    from nova.teacher import TeacherClient

    div = dict(ds.get("diversify", {}))
    print("[factory] stage: diversify")
    teacher = None
    if not div.get("dry_run"):
        teacher = TeacherClient.from_config(dict(div.get("teacher", {})))
        if not teacher.enabled:
            print("  no teacher API key -> dry-run plan only")
    stats = diversify(
        ds["base_parts_glob"],
        str(out_dir / "diversified.jsonl"),
        teacher=teacher,
        min_template_count=int(div.get("min_template_count", 1000)),
        limit=int(div.get("limit", 0)),
        dry_run=bool(div.get("dry_run", False)) or teacher is None
        or not getattr(teacher, "enabled", False),
        resume_from=str(div.get("resume_from", "")),
        cost_per_mtok_in=float(div.get("cost_per_mtok_in", 0.27)),
        cost_per_mtok_out=float(div.get("cost_per_mtok_out", 1.10)),
        max_tokens=int(div.get("max_tokens", 600)),
    )
    print(f"  collapsed={stats['collapsed_records']} "
          f"rewritten={stats['rewritten']} est_cost=${stats['est_cost_usd']}")
    return stats


def run_ingest(ds: dict, out_dir: Path) -> dict:
    from nova.datasets.commoncrawl import ingest, list_crawls

    cc = dict(ds.get("commoncrawl", {}))
    print("[factory] stage: ingest (Common Crawl)")
    crawl = cc.get("crawl_id") or list_crawls()[0]["id"]
    stats = ingest(
        crawl,
        list(cc.get("domains", [])),
        str(out_dir / "commoncrawl.jsonl"),
        max_records=int(cc.get("max_records", 100_000)),
        language=str(cc.get("language", "en")),
        min_chars=int(cc.get("min_chars", 200)),
        checkpoint_path=str(out_dir / "cc-checkpoint.json"),
    )
    print(f"  emitted={stats['emitted']}")
    return stats


def run_mix(ds: dict, out_dir: Path, stage_paths: dict) -> dict:
    from nova.datasets.mix import build_base_hash_store, mix

    mx = dict(ds.get("mixer", {}))
    print("[factory] stage: mix")
    db_path = mx.get("base_hash_db", "data/base_hashes.sqlite")
    if not Path(db_path).exists():
        print("  building base hash store (one-time)...")
        print(" ", build_base_hash_store(ds["base_parts_glob"], db_path))

    sources = []
    for name, key in (("diversified", "diversified_path"),
                      ("commoncrawl", "commoncrawl_path")):
        p = stage_paths.get(key)
        if p and Path(p).exists():
            sources.append({"name": name, "pattern": p})
    for extra in mx.get("extra_sources", []) or []:
        sources.append(extra)
    if not sources:
        print("  no new sources produced by earlier stages; nothing to mix")
        return {"skipped": True}

    stats = mix(
        sources=sources,
        base_db=db_path,
        out_dir=mx.get("output_dir", f"data/dataset-v{ds.get('version', '8')}"),
        version=str(ds.get("version", "8")),
        category_weights=mx.get("category_weights"),
        split_ratios=mx.get("split_ratios"),
        near_dup_threshold_bits=int(mx.get("near_dup_threshold_bits", 8)),
        enable_near_dedup=bool(mx.get("enable_near_dedup", True)),
        records_per_part=int(mx.get("records_per_part", 500_000)),
        seed=int(mx.get("seed", 42)),
        min_text_chars=int(mx.get("min_text_chars", 50)),
    )
    print(f"  accepted={stats['accepted']} splits={stats['splits']}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NOVA dataset factory: profile -> diversify -> ingest -> mix.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--workdir", default="data/factory",
                        help="Stage outputs live here")
    args = parser.parse_args(argv)

    ds, _ = _load_datasets_config(args.config)
    if not ds.get("enabled", True):
        print("datasets factory disabled in config")
        return 0
    if not ds.get("base_parts_glob"):
        print("error: datasets.base_parts_glob is not set", file=sys.stderr)
        return 2

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {}
    stage_paths: dict = {}

    stages = ds.get("stages", {})
    if stages.get("profile", True):
        summary["profile"] = run_profile(ds, workdir)
    if stages.get("diversify", True):
        summary["diversify"] = run_diversify(ds, workdir)
        stage_paths["diversified_path"] = str(workdir / "diversified.jsonl")
    if stages.get("ingest", True):
        summary["ingest"] = run_ingest(ds, workdir)
        stage_paths["commoncrawl_path"] = str(workdir / "commoncrawl.jsonl")
    if stages.get("mix", True):
        summary["mix"] = run_mix(ds, workdir, stage_paths)

    summary["seconds"] = round(time.time() - t0, 1)
    (workdir / "factory-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[factory] done in {summary['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
