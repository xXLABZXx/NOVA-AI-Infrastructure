"""Evaluation harness: join predictions with expected answers, score, report.

Predictions: ``[{"id": ..., "output": ...}, ...]``.
Expected:    ``[{"id": ..., "expected": ...}, ...]``.
Challenge ids (optional): a set of ids reported separately under
``by_split.challenge`` so held-out/challenge performance is visible on
its own.

Run as a module: ``python -m nova.evaluation.harness --predictions ...
--expected ... --report ...``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nova.evaluation.metrics import exact_match


def _aggregate(items: list[dict]) -> dict:
    total = len(items)
    correct = sum(1 for item in items if item.get("exact_match"))
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def run_evaluation(
    predictions: list[dict],
    expected: list[dict],
    challenge_ids: set[str] | None = None,
) -> dict:
    """Score predictions against expected answers.

    Returns ``{total, correct, accuracy, by_split: {standard, challenge},
    items, unmatched}``. Predictions with no matching expected id are
    listed in ``items`` with a note and excluded from the aggregates.
    """
    challenge_ids = set(challenge_ids or [])
    expected_by_id = {item["id"]: item.get("expected") for item in expected}

    items: list[dict] = []
    for pred in predictions:
        pid = pred.get("id")
        output = pred.get("output")
        if pid not in expected_by_id:
            items.append(
                {
                    "id": pid,
                    "output": output,
                    "expected": None,
                    "exact_match": False,
                    "note": "no expected answer for this id",
                }
            )
            continue
        items.append(
            {
                "id": pid,
                "output": output,
                "expected": expected_by_id[pid],
                "exact_match": exact_match(output, expected_by_id[pid]),
            }
        )

    matched = [i for i in items if "note" not in i]
    standard = [i for i in matched if i["id"] not in challenge_ids]
    challenge = [i for i in matched if i["id"] in challenge_ids]
    overall = _aggregate(matched)
    return {
        "total": overall["total"],
        "correct": overall["correct"],
        "accuracy": overall["accuracy"],
        "by_split": {
            "standard": _aggregate(standard),
            "challenge": _aggregate(challenge),
        },
        "items": items,
        "unmatched": len(items) - len(matched),
    }


def write_report(results: dict, path: str | Path) -> Path:
    """Write the evaluation results dict as pretty JSON; returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out


def _read_records(path: str | Path) -> list[dict]:
    """Read a JSONL file (one JSON object per line); also accepts a JSON array."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array or JSONL")
        return data
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}")
    return records


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m nova.evaluation.harness``."""
    parser = argparse.ArgumentParser(
        prog="nova.evaluation.harness",
        description="Score prediction JSON against expected answers.",
    )
    parser.add_argument("--predictions", required=True,
                        help="JSONL file: one {id, output} per line (JSON array also accepted)")
    parser.add_argument("--expected", required=True,
                        help="JSONL file: one {id, expected} per line (JSON array also accepted)")
    parser.add_argument("--report", required=True, help="output report path (JSON)")
    parser.add_argument("--challenge-ids", default=None,
                        help="optional file with one challenge id per line")
    args = parser.parse_args(argv)

    predictions = _read_records(args.predictions)
    expected = _read_records(args.expected)
    challenge_ids: set[str] | None = None
    if args.challenge_ids:
        challenge_ids = {
            line.strip()
            for line in Path(args.challenge_ids).read_text(
                encoding="utf-8").splitlines()
            if line.strip()
        }

    results = run_evaluation(predictions, expected, challenge_ids)
    write_report(results, args.report)
    print(json.dumps(
        {"accuracy": results["accuracy"], "by_split": results["by_split"],
         "report": args.report},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
