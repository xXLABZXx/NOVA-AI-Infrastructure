"""Teacher-driven answer diversification for template-collapsed categories.

The base dataset has ~52 unique answers across 17.4M records. This module
rewrites collapsed answers via the configured teacher model (DeepSeek by
default), preserving the original prompt and id, and stamping provenance.

Two streaming passes over the parts:
  1. discover templates: (category, answer_sha256) -> count  [tiny in RAM]
  2. rewrite records whose answer is a collapsed template.

CLI: ``python -m nova.datasets.diversify --parts 'glob' --out out.jsonl
[--limit N] [--dry-run] [--resume-from ID] [--config cfg]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from nova.datasets.base import PartStats, iter_jsonl_parts, write_jsonl

# Rotation of rewrite instructions, seeded per record -> forced diversity.
REWRITE_STYLES = [
    "Rewrite the answer completely differently while staying correct and on-topic. Use a concrete example.",
    "Rewrite the answer as a numbered step-by-step procedure with different wording and structure.",
    "Rewrite the answer conversationally, as if explaining to a colleague, with fresh phrasing.",
    "Rewrite the answer concisely in a different structure: lead with the conclusion, then the reasoning.",
    "Rewrite the answer with an analogy and varied vocabulary; do not reuse the original sentence patterns.",
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def discover_templates(pattern: str, *, min_count: int = 2) -> dict:
    """Pass 1: map (category, answer_sha256) -> {answer, count}.

    Only templates with count >= min_count are returned. Memory is bounded
    by the number of *unique* answers (tiny for a collapsed dataset).
    """
    stats = PartStats()
    counts: Counter = Counter()
    texts: dict = {}
    for rec in iter_jsonl_parts(pattern, stats=stats):
        key = (str(rec.get("category", "unknown")),
               _sha256(str(rec.get("answer", ""))))
        counts[key] += 1
        if key not in texts:
            texts[key] = str(rec.get("answer", ""))[:4000]
    return {
        key: {"answer": texts[key], "count": counts[key]}
        for key in counts
        if counts[key] >= min_count
    }


def _rewrite_prompt(record: dict, template: str, style_idx: int) -> str:
    style = REWRITE_STYLES[style_idx % len(REWRITE_STYLES)]
    return (
        "You are improving a training dataset. Given the QUESTION and the "
        "ORIGINAL ANSWER below, produce a NEW answer.\n\n"
        f"QUESTION:\n{record.get('prompt', '')}\n\n"
        f"ORIGINAL ANSWER:\n{template}\n\n"
        f"Instruction: {style}\n"
        "Rules: stay factually consistent with the original; do not repeat the "
        "question; do not mention these instructions; output only the new answer."
    )


def validate_rewrite(new_answer: str, template: str, prompt: str,
                     *, min_chars: int = 20, max_chars: int = 8000) -> tuple[bool, str]:
    """Return (ok, reason)."""
    text = (new_answer or "").strip()
    if not text:
        return False, "empty"
    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"
    norm = lambda s: " ".join(s.lower().split())
    if norm(text) == norm(template):
        return False, "identical_to_template"
    if prompt and norm(prompt)[:200] in norm(text)[:400] and len(text) < len(prompt):
        return False, "prompt_echo"
    return True, "ok"


def diversify(
    pattern: str,
    out_path: str,
    *,
    teacher=None,
    templates: dict | None = None,
    min_template_count: int = 1000,
    limit: int = 0,
    dry_run: bool = False,
    resume_from: str = "",
    cost_per_mtok_in: float = 0.27,
    cost_per_mtok_out: float = 1.10,
    max_tokens: int = 600,
    sleep_seconds: float = 0.0,
) -> dict:
    """Rewrite collapsed answers. Returns a stats dict.

    With ``dry_run=True`` (or a disabled teacher) no API calls are made:
    returns the plan + cost estimate instead.
    """
    stats = PartStats()
    templates = templates if templates is not None else discover_templates(
        pattern, min_count=min_template_count)
    template_keys = set(templates)
    total = collapsed = rewritten = skipped = failed = 0
    resumed = not resume_from
    est_prompt_tokens = 0
    est_completion_tokens = 0
    out_records: list[dict] = []

    def flush():
        if out_records:
            write_jsonl(out_path, iter(out_records), append=True)
            out_records.clear()

    for rec in iter_jsonl_parts(pattern, stats=stats):
        total += 1
        rid = str(rec.get("id", ""))
        if not resumed:
            if rid == resume_from:
                resumed = True
            continue
        key = (str(rec.get("category", "unknown")),
               _sha256(str(rec.get("answer", ""))))
        if key not in template_keys:
            skipped += 1
            continue
        collapsed += 1
        if limit and (rewritten >= limit or
                      (dry_run and collapsed >= limit)):
            break
        template = templates[key]["answer"]
        style_idx = int(_sha256(rid), 16) % len(REWRITE_STYLES)
        prompt_text = _rewrite_prompt(rec, template, style_idx)
        est_prompt_tokens += len(prompt_text) // 4
        est_completion_tokens += max_tokens // 2
        if dry_run or teacher is None or not getattr(teacher, "enabled", False):
            continue
        try:
            result = teacher.complete(prompt_text, max_tokens=max_tokens)
        except Exception as exc:  # budget exhausted, network, ...
            failed += 1
            out_records.append({
                "id": rid, "status": "failed",
                "reason": f"teacher_error: {type(exc).__name__}",
            })
            continue
        new_answer = (result.get("content") or "").strip()
        ok, reason = validate_rewrite(new_answer, template, str(rec.get("prompt", "")))
        if not ok:
            failed += 1
            out_records.append({"id": rid, "status": "rejected",
                                "reason": f"validation:{reason}"})
            continue
        new_rec = dict(rec)
        new_rec["answer"] = new_answer
        new_rec["provenance"] = dict(rec.get("provenance") or {})
        new_rec["provenance"].update({
            "rewritten_by": "diversify",
            "rewrite_model": getattr(teacher, "model", "unknown"),
            "original_answer_sha256": key[1],
            "rewrite_style": style_idx,
        })
        out_records.append(new_rec)
        rewritten += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
        if len(out_records) >= 500:
            flush()
    flush()

    est_cost = (est_prompt_tokens / 1e6) * cost_per_mtok_in + \
               (est_completion_tokens / 1e6) * cost_per_mtok_out
    return {
        "total_scanned": total,
        "collapsed_records": collapsed,
        "rewritten": rewritten,
        "skipped_non_template": skipped,
        "failed_or_rejected": failed,
        "malformed_lines": stats.malformed,
        "dry_run": dry_run or teacher is None or not getattr(teacher, "enabled", False),
        "est_prompt_tokens": est_prompt_tokens,
        "est_completion_tokens": est_completion_tokens,
        "est_cost_usd": round(est_cost, 2),
        "out_path": out_path,
    }


def main(argv: list[str] | None = None) -> int:
    from nova.config import load_config
    from nova.teacher import TeacherClient

    parser = argparse.ArgumentParser(
        description="Diversify collapsed answers via the teacher model.")
    parser.add_argument("--parts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max records to rewrite (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan + cost estimate, no API calls")
    parser.add_argument("--resume-from", default="",
                        help="Record id to resume after")
    parser.add_argument("--min-template-count", type=int, default=1000)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ds = cfg.section("datasets") if hasattr(cfg, "section") else {}
    div = dict(ds.get("diversify", {})) if isinstance(ds, dict) else {}
    teacher = TeacherClient.from_config(
        dict(div.get("teacher", {}))) if not args.dry_run else None
    if teacher is not None and not teacher.enabled:
        print("teacher disabled (no API key) -> running as dry-run plan")
        args.dry_run = True

    stats = diversify(
        args.parts, args.out,
        teacher=teacher,
        min_template_count=args.min_template_count,
        limit=args.limit,
        dry_run=args.dry_run,
        resume_from=args.resume_from,
        cost_per_mtok_in=float(div.get("cost_per_mtok_in", 0.27)),
        cost_per_mtok_out=float(div.get("cost_per_mtok_out", 1.10)),
        max_tokens=int(div.get("max_tokens", 600)),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
