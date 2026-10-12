"""Offline quality checks for training candidates.

``review_candidate`` runs purely in Python (no teacher API needed) and is
meant as a cheap first pass before any expensive teacher review.
``generate_eval_questions`` and ``quality_report`` build on top of it.
"""

from __future__ import annotations

import hashlib
import json
import re

# Patterns that suggest prompt-injection / instruction-hijacking attempts.
_INJECTION_PATTERNS = (
    r"ignore (all )?previous instructions",
    r"ignore (all )?prior instructions",
    r"disregard (all )?(previous|prior) instructions",
    r"do anything now",
    r"\bjailbreak\b",
    r"<script",
    r"\bsystem\s*:",
)

_MIN_TEXT_CHARS = 10
_MAX_TEXT_CHARS = 100_000


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _security_scan(text: str) -> list[str]:
    """Optional hook into nova.security; falls back to the local patterns.

    The cross-package import is lazy and any failure is swallowed, so this
    module never depends on nova.security being installed.
    """
    try:
        from nova import security  # lazy, optional

        scan = getattr(security, "scan_text", None)
        if callable(scan):
            findings = scan(text)
            return [str(f) for f in findings] if findings else []
    except Exception:
        pass
    return []


def _injection_hits(text: str) -> list[str]:
    hits = []
    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            hits.append(pattern)
    return hits


def review_candidate(
    example: dict,
    approved_examples: list[dict] | None = None,
) -> dict:
    """Run cheap offline checks on a training candidate.

    Returns ``{example_id, checks, issues, verdict}`` where ``verdict`` is
    ``"pass"``, ``"needs_review"``, or ``"reject"``.
    """
    checks: dict[str, bool] = {}
    issues: list[str] = []

    example_id = example.get("id")
    text = example.get("text")
    checks["has_required_fields"] = bool(example_id) and isinstance(text, str) and bool(text.strip())
    if not checks["has_required_fields"]:
        issues.append("missing required fields: 'id' and non-empty 'text'")

    body = text if isinstance(text, str) else ""
    checks["text_length_ok"] = _MIN_TEXT_CHARS <= len(body) <= _MAX_TEXT_CHARS
    if isinstance(text, str) and text.strip() and not checks["text_length_ok"]:
        issues.append(
            f"text length {len(body)} outside [{_MIN_TEXT_CHARS}, {_MAX_TEXT_CHARS}]"
        )

    injection = _injection_hits(body) + _security_scan(body)
    checks["no_injection_flags"] = not injection
    for hit in injection:
        issues.append(f"possible injection pattern: {hit}")

    duplicate_of = None
    if approved_examples:
        seen = {
            hashlib.sha256(_normalize(e.get("text", "")).encode("utf-8")).hexdigest()
            for e in approved_examples
            if isinstance(e.get("text"), str)
        }
        digest = hashlib.sha256(_normalize(body).encode("utf-8")).hexdigest()
        if digest in seen:
            duplicate_of = True
            issues.append("duplicate of an already-approved example")
    checks["not_duplicate"] = duplicate_of is not True

    if not checks["has_required_fields"] or not checks["no_injection_flags"]:
        verdict = "reject"
    elif issues:
        verdict = "needs_review"
    else:
        verdict = "pass"
    return {
        "example_id": example_id,
        "checks": checks,
        "issues": issues,
        "verdict": verdict,
    }


def _extract_json_array(text: str):
    """Best-effort extraction of a JSON array from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("no JSON array found in teacher response")


def generate_eval_questions(
    client, topic: str, n: int = 5
) -> list[dict]:
    """Ask the teacher model for ``n`` evaluation questions about ``topic``.

    Returns a list of ``{"question": ..., "topic": ...}`` dicts. When the
    client is disabled (no API key) no network call is made and a single
    note dict is returned instead.
    """
    if not client.enabled:
        return [
            {
                "note": "teacher disabled: no API key configured; "
                        "no questions generated",
                "topic": topic,
                "questions": [],
            }
        ]
    prompt = (
        f"Generate {n} evaluation questions about the following topic. "
        "Reply with ONLY a JSON array of objects, each with a single key "
        f"'question'.\nTopic: {topic}"
    )
    response = client.complete(prompt)
    content = response.get("content", "") if isinstance(response, dict) else ""
    try:
        data = _extract_json_array(content)
    except (ValueError, json.JSONDecodeError):
        return [
            {
                "note": "could not parse teacher response as a question list",
                "topic": topic,
                "raw": content[:500],
            }
        ]
    if not isinstance(data, list):
        return [
            {
                "note": "teacher response was not a list",
                "topic": topic,
                "raw": content[:500],
            }
        ]
    questions = []
    for item in data[:n]:
        if isinstance(item, dict) and item.get("question"):
            questions.append({"question": str(item["question"]), "topic": topic})
        elif isinstance(item, str):
            questions.append({"question": item, "topic": topic})
    return questions


def quality_report(examples: list[dict], reviews: list[dict]) -> dict:
    """Aggregate review verdicts into a summary report."""
    verdict_counts: dict[str, int] = {}
    issue_histogram: dict[str, int] = {}
    for review in reviews:
        verdict = review.get("verdict", "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        for issue in review.get("issues", []):
            issue_histogram[issue] = issue_histogram.get(issue, 0) + 1
    total = len(reviews)
    passes = verdict_counts.get("pass", 0)
    return {
        "total_examples": len(examples),
        "total_reviews": total,
        "verdicts": verdict_counts,
        "issues": issue_histogram,
        "pass_rate": (passes / total) if total else 0.0,
    }
