"""Curation pipeline: raw -> cleaned -> approved / rejected.

Reads every ``*.jsonl`` in ``raw_dir`` (raw files are only ever read,
never modified), enriches each document with quality scores, routes it
to the approved or rejected dataset, records every decision in the
change log, and writes a content manifest for the approved output.

Run as a module: ``python -m nova.curation.pipeline --config ...``
(backing ``scripts/curate.sh``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from nova.curation.changelog import ChangeLog
from nova.curation.scoring import quality_score, source_reliability
from nova.curation.versions import write_manifest


def _notify_monitoring(event: str, payload: dict) -> None:
    """Best-effort hook into nova.monitoring.

    The import is lazy and any failure (module missing, API mismatch) is
    silently ignored so curation never depends on the monitoring package.
    """
    try:
        monitoring = importlib.import_module("nova.monitoring")
        emit = getattr(monitoring, "emit_event", None)
        if callable(emit):
            emit(event, payload)
    except Exception:
        return


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rejection_reason(
    q: float, r: float, min_quality_score: float, min_source_reliability: float
) -> str:
    reasons = []
    if q < min_quality_score:
        reasons.append(
            f"quality_score {q:.3f} below threshold {min_quality_score}"
        )
    if r < min_source_reliability:
        reasons.append(
            f"source_reliability {r:.3f} below threshold {min_source_reliability}"
        )
    return "; ".join(reasons) or "did not meet curation thresholds"


def run_pipeline(
    *,
    raw_dir: str | Path,
    cleaned_dir: str | Path,
    approved_dir: str | Path,
    rejected_dir: str | Path,
    changelog_path: str | Path,
    min_quality_score: float = 0.35,
    min_source_reliability: float = 0.2,
    require_human_approval: bool = True,
    version: str = "v1",
) -> dict:
    """Run the curation pipeline and return run statistics.

    Stats dict: ``{total, cleaned, approved, rejected, avg_quality}``.
    Every document gets exactly one change-log entry (``action`` is
    ``"clean"`` or ``"reject"``). Raw input files are never modified.
    """
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    approved_dir = Path(approved_dir)
    rejected_dir = Path(rejected_dir)
    for d in (cleaned_dir, approved_dir, rejected_dir):
        d.mkdir(parents=True, exist_ok=True)

    changelog = ChangeLog(changelog_path)
    raw_files = sorted(raw_dir.glob("*.jsonl"))
    if not raw_files:
        raise FileNotFoundError(f"no *.jsonl files found in {raw_dir}")

    cleaned: list[dict] = []
    approved: list[dict] = []
    rejected: list[dict] = []
    quality_sum = 0.0
    total = 0

    for raw_file in raw_files:
        # Raw files are opened read-only; nothing here writes to raw_dir.
        with open(raw_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                total += 1

                q = quality_score(doc)
                r = source_reliability(
                    doc.get("url") or doc.get("source") or ""
                )
                quality_sum += q
                doc_id = str(doc.get("id", f"line-{total}"))

                enriched = dict(doc)  # never mutate the parsed raw dict in place
                enriched["quality_score"] = q
                enriched["source_reliability"] = r
                enriched["curated_at"] = _utcnow()
                cleaned.append(enriched)

                passed = q >= min_quality_score and r >= min_source_reliability
                validation = {
                    "min_quality_score": min_quality_score,
                    "min_source_reliability": min_source_reliability,
                    "quality_score": q,
                    "source_reliability": r,
                }
                if passed:
                    entry_id = changelog.propose(
                        record_id=doc_id,
                        proposed_change={
                            "action": "clean",
                            "quality_score": q,
                            "source_reliability": r,
                        },
                        reason=f"curation pipeline {version}: scores above thresholds",
                        validation=validation,
                    )
                    if require_human_approval:
                        pass  # entry stays pending; doc waits in cleaned/
                    else:
                        approved.append(enriched)
                        changelog.approve(entry_id, approver="pipeline")
                else:
                    reason = _rejection_reason(
                        q, r, min_quality_score, min_source_reliability
                    )
                    rejected_doc = dict(enriched)
                    rejected_doc["rejection_reason"] = reason
                    rejected.append(rejected_doc)
                    entry_id = changelog.propose(
                        record_id=doc_id,
                        proposed_change={
                            "action": "reject",
                            "quality_score": q,
                            "source_reliability": r,
                            "rejection_reason": reason,
                        },
                        reason=f"curation pipeline {version}: {reason}",
                        validation=validation,
                    )
                    changelog.approve(entry_id, approver="pipeline")

    cleaned_path = cleaned_dir / "all.jsonl"
    with open(cleaned_path, "w", encoding="utf-8") as fh:
        for doc in cleaned:
            fh.write(json.dumps(doc) + "\n")

    approved_path = approved_dir / f"dataset-{version}.jsonl"
    if approved:
        with open(approved_path, "w", encoding="utf-8") as fh:
            for doc in approved:
                fh.write(json.dumps(doc) + "\n")

    rejected_path = rejected_dir / "rejected.jsonl"
    with open(rejected_path, "w", encoding="utf-8") as fh:
        for doc in rejected:
            fh.write(json.dumps(doc) + "\n")

    manifest_path = write_manifest(approved_dir, version)

    stats = {
        "total": total,
        "cleaned": len(cleaned),
        "approved": len(approved),
        "rejected": len(rejected),
        "avg_quality": (quality_sum / total) if total else 0.0,
        "manifest": str(manifest_path),
    }
    _notify_monitoring(
        "curation.pipeline.completed",
        {"version": version, **{k: v for k, v in stats.items() if k != "manifest"}},
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m nova.curation.pipeline``."""
    parser = argparse.ArgumentParser(
        prog="nova.curation.pipeline",
        description="Curate raw JSONL documents into approved/rejected datasets.",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--raw", required=True, help="directory of raw *.jsonl")
    parser.add_argument("--cleaned", required=True, help="output dir for cleaned docs")
    parser.add_argument("--approved", required=True, help="output dir for approved dataset")
    parser.add_argument("--rejected", required=True, help="output dir for rejected docs")
    parser.add_argument("--version", default="v1", help="dataset version label")
    args = parser.parse_args(argv)

    # Lazy config import: the pipeline works without nova.config installed.
    min_quality_score = 0.35
    min_source_reliability = 0.2
    require_human_approval = True
    if args.config:
        from nova.config import load_config  # lazy

        cfg = load_config(args.config).section("curation")
        min_quality_score = float(cfg.get("min_quality_score", min_quality_score))
        min_source_reliability = float(
            cfg.get("min_source_reliability", min_source_reliability)
        )
        require_human_approval = bool(
            cfg.get("require_human_approval", require_human_approval)
        )

    stats = run_pipeline(
        raw_dir=args.raw,
        cleaned_dir=args.cleaned,
        approved_dir=args.approved,
        rejected_dir=args.rejected,
        changelog_path=Path(args.approved).parent / "changelog.jsonl",
        min_quality_score=min_quality_score,
        min_source_reliability=min_source_reliability,
        require_human_approval=require_human_approval,
        version=args.version,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
