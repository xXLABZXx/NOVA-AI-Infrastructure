"""Scan text and files for leaked secrets (API keys, tokens, private keys).

Findings never contain the secret itself: excerpts show the first four
characters of the matched span followed by ``***``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SecretFinding", "SECRET_PATTERNS", "scan_text_for_secrets",
           "scan_file_for_secrets"]


@dataclass(frozen=True)
class SecretFinding:
    """One detected secret occurrence."""

    pattern_name: str
    line_no: int  #: 1-based line number
    excerpt_redacted: str  #: surrounding line with the secret redacted


#: Named, compiled secret detectors.
SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret_key": re.compile(
        r"aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*['\"]?"
        r"[A-Za-z0-9/+=]{20,}['\"]?",
        re.IGNORECASE,
    ),
    "github_token": re.compile(
        r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}"
    ),
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "generic_api_key": re.compile(
        r"(api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9_\-.]{8,}['\"]?",
        re.IGNORECASE,
    ),
    "private_key": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "bearer_token": re.compile(
        r"bearer\s+[A-Za-z0-9_\-.~+/=]{10,}", re.IGNORECASE
    ),
}

#: Max characters kept in a redacted excerpt.
_MAX_EXCERPT_CHARS = 200


def _redact_match(line: str, match: re.Match) -> str:
    """Replace the matched span with its first 4 chars + ``***``."""
    span = match.group()
    return line[: match.start()] + span[:4] + "***" + line[match.end():]


def scan_text_for_secrets(text: str) -> list[SecretFinding]:
    """Return a :class:`SecretFinding` for every secret-looking span.

    ``line_no`` is 1-based; ``excerpt_redacted`` never contains the full
    secret value.
    """
    if not isinstance(text, str):
        raise TypeError(f"scan_text_for_secrets expects str, got {type(text).__name__}")
    findings: list[SecretFinding] = []
    seen: set[tuple[str, int, int]] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern_name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(line):
                key = (pattern_name, line_no, match.start())
                if key in seen:
                    continue
                seen.add(key)
                excerpt = _redact_match(line, match).strip()
                findings.append(
                    SecretFinding(
                        pattern_name=pattern_name,
                        line_no=line_no,
                        excerpt_redacted=excerpt[:_MAX_EXCERPT_CHARS],
                    )
                )
    return findings


def scan_file_for_secrets(path: str | Path) -> list[SecretFinding]:
    """Scan a file's text content for secrets (binary-safe decode)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return scan_text_for_secrets(text)
