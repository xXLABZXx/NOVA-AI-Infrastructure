"""Dataset profiler: stream the base parts and report collapse/diversity.

Detects the template-collapse pathology: few unique answers across millions
of records. Output is a JSON report; the CLI never loads data into RAM.

CLI: ``python -m nova.datasets.profile --parts 'glob' --out report.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from nova.datasets.base import PartStats, iter_jsonl_parts


def _percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * pct / 100))
    return sorted_vals[idx]


def profile_parts(
    pattern: str,
    *,
    max_answer_samples: int = 200,
    length_sample_every: int = 100,
) -> dict:
    """Stream ``pattern`` and build the quality report."""
    stats = PartStats()
    total = 0
    per_category: Counter = Counter()
    synthetic = 0
    prompt_hashes: set[str] = set()
    answer_counts: Counter = Counter()
    answer_samples: dict[str, str] = {}
    prompt_len_samples: list[int] = []
    answer_len_samples: list[int] = []

    def h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    for rec in iter_jsonl_parts(pattern, stats=stats):
        total += 1
        cat = str(rec.get("category", "unknown"))
        per_category[cat] += 1
        if rec.get("synthetic"):
            synthetic += 1
        prompt = str(rec.get("prompt", ""))
        answer = str(rec.get("answer", ""))
        prompt_hashes.add(h(prompt))
        ah = h(answer)
        answer_counts[ah] += 1
        if len(answer_samples) < max_answer_samples and ah not in answer_samples:
            answer_samples[ah] = answer[:500]
        if total % length_sample_every == 0:
            prompt_len_samples.append(len(prompt))
            answer_len_samples.append(len(answer))

    prompt_len_samples.sort()
    answer_len_samples.sort()
    unique_answers = len(answer_counts)
    top_answers = [
        {"count": c, "sample": answer_samples.get(ah, "")[:200]}
        for ah, c in answer_counts.most_common(10)
    ]
    return {
        "total_records": total,
        "malformed_lines": stats.malformed,
        "files_read": stats.files,
        "per_category": dict(per_category),
        "synthetic_ratio": round(synthetic / total, 4) if total else 0.0,
        "unique_prompts": len(prompt_hashes),
        "prompt_diversity": round(len(prompt_hashes) / total, 4) if total else 0.0,
        "unique_answers": unique_answers,
        "answer_collapse_ratio": round(1 - unique_answers / total, 4) if total else 0.0,
        "prompt_len_chars": {
            "p50": _percentile(prompt_len_samples, 50),
            "p99": _percentile(prompt_len_samples, 99),
        },
        "answer_len_chars": {
            "p50": _percentile(answer_len_samples, 50),
            "p99": _percentile(answer_len_samples, 99),
        },
        "top_answers": top_answers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile a JSONL dataset (streaming, low RAM).")
    parser.add_argument("--parts", required=True,
                        help="Glob for parts, or path.zip / path.zip:member-glob")
    parser.add_argument("--out", default="",
                        help="Write JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    report = profile_parts(args.parts)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}: {report['total_records']} records, "
              f"collapse_ratio={report['answer_collapse_ratio']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
