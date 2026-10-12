"""Structured event logging with automatic secret redaction.

``EventLogger.log(event, **fields)`` appends one JSON record per line to
``<log_dir>/<name>.jsonl``. Before writing, fields are scrubbed:

* any field whose key matches (case-insensitively)
  ``.*(api[_-]?key|token|secret|password|passwd|auth|bearer|private[_-]?key).*``
  has its value replaced with ``"[REDACTED]"``;
* any string value that matches a known secret pattern (see
  :mod:`nova.security.secret_scan`) or looks like an ``ENV_VAR=secret``
  assignment is replaced with ``"[REDACTED]"``.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.security.secret_scan import SECRET_PATTERNS

__all__ = ["EventLogger", "get_logger"]

REDACTED = "[REDACTED]"

#: Field names that always trigger redaction (case-insensitive).
_SENSITIVE_KEY_RE = re.compile(
    r".*(api[_-]?key|token|secret|password|passwd|auth|bearer|private[_-]?key).*",
    re.IGNORECASE,
)

#: Strings like ``DEEPSEEK_API_KEY=sk-...`` or ``TOKEN: abc123``.
_ENV_ASSIGNMENT_RE = re.compile(r"^\s*[A-Z][A-Z0-9_]{2,}\s*[:=]\s*\S")


def _string_holds_secret(value: str) -> bool:
    """True when a string value looks like it contains a secret."""
    if _ENV_ASSIGNMENT_RE.search(value):
        return True
    return any(pattern.search(value) for pattern in SECRET_PATTERNS.values())


def _redact(obj: Any) -> Any:
    """Recursively redact secrets in dicts, lists, tuples, and strings."""
    if isinstance(obj, dict):
        cleaned: dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key):
                cleaned[key] = REDACTED
            else:
                cleaned[key] = _redact(value)
        return cleaned
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_redact(item) for item in obj)
    if isinstance(obj, str) and _string_holds_secret(obj):
        return REDACTED
    return obj


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLogger:
    """Append-only JSONL event logger with secret redaction.

    Thread-safe for concurrent :meth:`log` calls (single lock around
    each append).
    """

    def __init__(self, log_dir: str | Path, name: str = "nova") -> None:
        self.log_dir = Path(log_dir)
        self.name = name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{name}.jsonl"
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, section: dict) -> "EventLogger":
        """Build from a config mapping, e.g. ``{"log_dir": "logs", "name": "nova"}``."""
        section = section or {}
        return cls(
            log_dir=section.get("log_dir", "logs"),
            name=section.get("name", "nova"),
        )

    def log(self, event: str, **fields: Any) -> dict:
        """Append one redacted JSON record and return the record written."""
        record: dict[str, Any] = {"timestamp": _utc_now_iso(), "event": event}
        record.update(_redact(fields))
        line = json.dumps(record, ensure_ascii=True, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record


def get_logger(log_dir: str | Path, name: str = "nova") -> EventLogger:
    """Return an :class:`EventLogger` writing to ``<log_dir>/<name>.jsonl``."""
    return EventLogger(log_dir, name=name)
