"""Text cleaning for ingested documents.

Pure functions: ``normalize_text``, ``strip_boilerplate``, ``detect_language``,
``filter_by_language`` and ``clean_document`` (which never mutates its input).
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "normalize_text",
    "strip_boilerplate",
    "detect_language",
    "filter_by_language",
    "clean_document",
]

#: Lines matching any of these look like chrome, not content.
_BOILERPLATE_PATTERNS = [
    re.compile(r"copyright|©|®|all rights reserved", re.IGNORECASE),
    re.compile(r"cookie|privacy policy|terms of (service|use)|gdpr", re.IGNORECASE),
    re.compile(r"subscribe|sign up|log in|login|create an account", re.IGNORECASE),
    re.compile(r"^\s*(home|menu|search|contact|about)(\s*[|›»/]\s*\w+){1,6}\s*$",
               re.IGNORECASE),
    re.compile(r"advertisement|sponsored", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    """NFKC-normalize, collapse all whitespace runs, strip ends."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_boilerplate(line: str) -> bool:
    return any(pattern.search(line) for pattern in _BOILERPLATE_PATTERNS)


def strip_boilerplate(text: str) -> str:
    """Drop boilerplate lines (nav, footer, copyright, cookie notices, ...).

    Also drops very short lines (< 3 words) unless most lines are short, in
    which case short lines are kept (e.g. poetry, code, or terse documents).
    """
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    short = sum(1 for ln in lines if len(ln.split()) < 3)
    mostly_short = short > len(lines) / 2

    kept = []
    for line in lines:
        if _looks_like_boilerplate(line):
            continue
        if not mostly_short and len(line.split()) < 3:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def detect_language(text: str) -> str:
    """Heuristic language id; lazily reuses the crawler's implementation."""
    try:
        from nova.crawlers.extract import detect_language as _detect
        return _detect(text)
    except Exception:
        # Minimal fallback so ingestion never hard-depends on crawlers.
        tokens = re.findall(r"[a-z]+", (text or "").lower())
        if len(" ".join(tokens)) < 50:
            return "unknown"
        english = {"the", "and", "of", "to", "in", "is", "that", "with"}
        hits = sum(1 for tok in tokens if tok in english)
        return "en" if hits >= 3 else "unknown"


def filter_by_language(docs: list[dict], allowed) -> list[dict]:
    """Keep documents whose language is in ``allowed``.

    Uses each doc's ``lang`` field when present, otherwise detects it.
    """
    allowed_set = {str(lang).lower() for lang in allowed}
    kept = []
    for doc in docs:
        lang = doc.get("lang") or detect_language(doc.get("text", ""))
        if str(lang).lower() in allowed_set:
            kept.append(doc)
    return kept


def clean_document(doc: dict) -> dict:
    """Return a cleaned *copy* of ``doc``; the input is never mutated.

    The copy's ``text`` is normalized and de-boilerplated, and its
    ``metadata`` gains ``{"cleaned": True}``.
    """
    cleaned = dict(doc)
    cleaned["text"] = strip_boilerplate(normalize_text(doc.get("text", "")))
    metadata = dict(doc.get("metadata") or {})
    metadata["cleaned"] = True
    cleaned["metadata"] = metadata
    return cleaned
