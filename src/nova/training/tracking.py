"""Experiment bookkeeping: loss logs and config snapshots.

Both are plain JSON/JSONL files; no training framework required.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class LossTracker:
    """Append-only JSONL log of per-step training/validation loss."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        step: int,
        train_loss: float,
        val_loss: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append one step record to the log."""
        record = {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "logged_at": _utcnow(),
        }
        if extra:
            record.update(dict(extra))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        """Read every logged step record, in order."""
        records: list[dict] = []
        if not self.path.exists():
            return records
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


class ExperimentSnapshot:
    """Persist/restore the exact config an experiment ran with."""

    @staticmethod
    def save(config, path: str | Path) -> Path:
        """Write ``{"config": ..., "saved_at": ...}`` JSON; returns the path."""
        payload = {
            "config": asdict(config),
            "saved_at": _utcnow(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def load(path: str | Path) -> dict:
        """Read a snapshot file back into a dict."""
        return json.loads(Path(path).read_text(encoding="utf-8"))
