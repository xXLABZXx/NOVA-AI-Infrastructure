"""Quality scoring and source reliability for the curation pipeline.

All functions here are pure and deterministic: same document in, same
score out. They use only the standard library so the curation stage can
run anywhere without heavy dependencies.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Hosts treated as highly reputable (score 0.9). Matched on the registrable
# domain or any subdomain of it.
_REPUTABLE_DOMAINS = (
    "wikipedia.org",
    "wikimedia.org",
    "arxiv.org",
    "docs.python.org",
    "python.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "jstor.org",
    "plos.org",
    "pubmed.ncbi.nlm.nih.gov",
    "acm.org",
    "ieee.org",
)

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens used for the lexical-diversity component."""
    return _TOKEN_RE.findall(text.lower())


def _repeated_line_ratio(text: str) -> float:
    """Fraction of non-empty lines that are duplicates (0.0 = all unique)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def quality_score(doc: dict) -> float:
    """Score a document's intrinsic text quality in [0, 1].

    Combines three signals:

    * normalized text length, saturating at 1000 characters (longer text
      carries more signal, with diminishing returns);
    * lexical diversity (unique tokens / total tokens);
    * a boilerplate penalty for line-level repetition.

    Very short texts (< 50 chars, or < 200 chars to a lesser degree) take
    an additional penalty, since they rarely contain usable training signal.

    Deterministic: identical input always yields an identical score.
    """
    text = doc.get("text") or ""
    if not isinstance(text, str):
        text = str(text)
    chars = len(text)

    length_score = min(chars / 1000.0, 1.0)

    toks = _tokens(text)
    diversity = (len(set(toks)) / len(toks)) if toks else 0.0

    repetition = _repeated_line_ratio(text)

    score = (
        0.5 * length_score
        + 0.3 * diversity
        + 0.2 * (1.0 - repetition)
    )
    if chars < 50:
        score -= 0.5
    elif chars < 200:
        score -= 0.15
    return max(0.0, min(1.0, score))


def _host(value: str) -> str:
    """Best-effort hostname extraction from a URL or bare host/source string.

    Returns "" when the value does not look like a URL or hostname at all
    (e.g. a bare source label), so it scores as unknown.
    """
    value = (value or "").strip().lower()
    if not value:
        return ""
    had_scheme = "://" in value or value.startswith("//")
    if not had_scheme:
        value = "//" + value
    try:
        host = urlparse(value).hostname or ""
    except ValueError:
        return ""
    # A bare label with no dot and no scheme is a source name, not a host.
    if not had_scheme and "." not in host:
        return ""
    return host


def source_reliability(url_or_source: str) -> float:
    """Score the reliability of a document's source in [0, 1].

    Tiered by host:

    * well-known reputable domains (wikipedia.org, arxiv.org,
      docs.python.org, ...) -> 0.9
    * .edu / .gov hosts -> 0.8
    * github.com -> 0.7
    * any other http(s) URL -> 0.5
    * unknown, empty, or non-URL source -> 0.3

    Deterministic.
    """
    host = _host(url_or_source or "")
    if not host:
        return 0.3
    for domain in _REPUTABLE_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return 0.9
    if host.endswith(".edu") or host.endswith(".gov"):
        return 0.8
    if host == "github.com" or host.endswith(".github.com"):
        return 0.7
    return 0.5
