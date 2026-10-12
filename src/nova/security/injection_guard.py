"""Prompt-injection guard for untrusted text.

``guard_text`` wraps untrusted input in ``<untrusted-text>`` delimiters,
detects instruction-like spans (instruction overrides, fake
``system:`` headers, secret-exfiltration requests, fake tool-call
markup), removes them, and reports what was found. The delimiters help a
downstream model treat the content as data, not instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["GuardResult", "guard_text", "UNTRUSTED_OPEN", "UNTRUSTED_CLOSE"]

#: Delimiters marking untrusted content in model-facing output.
UNTRUSTED_OPEN = "<untrusted-text>"
UNTRUSTED_CLOSE = "</untrusted-text>"

#: Ceiling for a single guarded document (matches security.max_input_chars).
DEFAULT_MAX_CHARS = 1_000_000


@dataclass
class GuardResult:
    """Outcome of guarding one piece of untrusted text."""

    safe_text: str  #: cleaned text wrapped in <untrusted-text> delimiters
    flagged: bool  #: True when any malicious span was detected
    reasons: list[str] = field(default_factory=list)  #: detection names, in order


# (reason, compiled pattern) pairs, evaluated in order.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b"
            r"[^.!?\n]{0,100}?\b(previous|prior|earlier|above|all)\b"
            r"[^.!?\n]{0,60}?\binstructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_injection",
        re.compile(r"^[ \t]*(system|developer|admin)[ \t]*:", re.IGNORECASE | re.MULTILINE),
    ),
    (
        "system_prompt_injection",
        re.compile(r"\[\s*(system|developer)\s*\]|<\s*(system|developer)\s*>", re.IGNORECASE),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(reveal|disclose|show|print|output|tell|give|share)\b"
            r"[^.!?\n]{0,80}?\b(your|the|my)\b"
            r"[^.!?\n]{0,60}?\b(api[\s_-]*keys?|secrets?|passwords?|tokens?|"
            r"system prompt|private keys?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fake_tool_call",
        re.compile(r"\[\s*/?\s*(tool|function)[_\s-]*calls?\b[^\]]*\]", re.IGNORECASE),
    ),
    (
        "fake_tool_call",
        re.compile(r"<\s*/?\s*(tool|function)[_\s-]*calls?\b[^>]*>", re.IGNORECASE),
    ),
    (
        "fake_tool_call",
        re.compile(r"\{\s*[\"']tool_calls?[\"']\s*:", re.IGNORECASE),
    ),
]

# Neutralize attacker-supplied delimiters so they cannot close ours early.
_DELIMITER_SPOOF = re.compile(r"</?untrusted-text>", re.IGNORECASE)


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove (start, end) spans from text, merging overlaps."""
    kept: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor:  # overlapping: extend the previous cut
            start = cursor
        if start >= end:
            continue
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    return "".join(kept)


def guard_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> GuardResult:
    """Guard untrusted text against prompt-injection patterns.

    Returns a :class:`GuardResult` whose ``safe_text`` is the cleaned
    text wrapped in ``<untrusted-text>`` delimiters. Malicious spans are
    removed and named in ``reasons``. Raises :class:`ValueError` when
    ``text`` exceeds ``max_chars``.
    """
    if not isinstance(text, str):
        raise TypeError(f"guard_text expects str, got {type(text).__name__}")
    if len(text) > max_chars:
        raise ValueError(
            f"input of {len(text)} chars exceeds limit of {max_chars}"
        )

    reasons: list[str] = []
    spans: list[tuple[int, int]] = []
    for reason, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
            if reason not in reasons:
                reasons.append(reason)

    cleaned = _strip_spans(text, spans)
    cleaned = _DELIMITER_SPOOF.sub("[untrusted-text]", cleaned)
    # Tidy leftover whitespace without reflowing the content.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    safe_text = f"{UNTRUSTED_OPEN}\n{cleaned}\n{UNTRUSTED_CLOSE}"
    return GuardResult(safe_text=safe_text, flagged=bool(reasons), reasons=reasons)
