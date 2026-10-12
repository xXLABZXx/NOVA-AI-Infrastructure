"""Append-only audit log: one JSON record per line (JSONL)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["write_audit_event"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_audit_event(
    log_path: str | Path, actor: str, action: str, **details: Any
) -> dict:
    """Append an audit event as one JSON line to ``log_path``.

    The record is ``{"timestamp": iso, "actor": ..., "action": ...,
    **details}``. Parent directories are created as needed. Returns the
    record that was written.
    """
    record: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "actor": actor,
        "action": action,
    }
    record.update(details)
    path = Path(log_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    return record
