"""Append-only human-in-the-loop change log for curated records.

Every proposed modification to a curated record is recorded as an entry
in a JSONL file before it takes effect. Entries start ``pending`` and
move to ``approved`` or ``rejected`` when a human (or an automated
pipeline acting explicitly) reviews them.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChangeLog:
    """JSONL-backed change log.

    Each entry is a dict with keys: ``entry_id``, ``timestamp``,
    ``record_id``, ``proposed_change``, ``reason``, ``validation``,
    ``approval_status`` (``pending`` | ``approved`` | ``rejected``),
    ``approver`` (``None`` while pending) and, for rejections,
    ``rejection_reason``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    # -- internal helpers -------------------------------------------------
    def _read_all(self) -> list[dict]:
        entries: list[dict] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _write_all(self, entries: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")

    def _append(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # -- public API --------------------------------------------------------
    def propose(
        self,
        record_id: str,
        proposed_change: dict,
        reason: str,
        validation: dict | None = None,
    ) -> str:
        """Record a proposed change; returns the new entry id (status pending)."""
        entry_id = uuid.uuid4().hex
        entry = {
            "entry_id": entry_id,
            "timestamp": _utcnow(),
            "record_id": record_id,
            "proposed_change": dict(proposed_change),
            "reason": reason,
            "validation": dict(validation) if validation else None,
            "approval_status": "pending",
            "approver": None,
        }
        self._append(entry)
        return entry_id

    def _set_status(
        self, entry_id: str, status: str, approver: str,
        rejection_reason: str | None = None,
    ) -> dict:
        entries = self._read_all()
        for entry in entries:
            if entry["entry_id"] == entry_id:
                if entry["approval_status"] != "pending":
                    raise ValueError(
                        f"entry {entry_id} already {entry['approval_status']}"
                    )
                entry["approval_status"] = status
                entry["approver"] = approver
                entry["reviewed_at"] = _utcnow()
                if rejection_reason is not None:
                    entry["rejection_reason"] = rejection_reason
                self._write_all(entries)
                return entry
        raise KeyError(f"no changelog entry {entry_id!r}")

    def approve(self, entry_id: str, approver: str) -> dict:
        """Mark a pending entry approved by ``approver``."""
        return self._set_status(entry_id, "approved", approver)

    def reject(self, entry_id: str, approver: str, reason: str) -> dict:
        """Mark a pending entry rejected by ``approver`` with ``reason``."""
        return self._set_status(entry_id, "rejected", approver,
                               rejection_reason=reason)

    def entries_for(self, record_id: str) -> list[dict]:
        """All entries (any status) recorded for ``record_id``."""
        return [e for e in self._read_all() if e.get("record_id") == record_id]

    def pending(self) -> list[dict]:
        """All entries still awaiting review."""
        return [e for e in self._read_all()
                if e.get("approval_status") == "pending"]
