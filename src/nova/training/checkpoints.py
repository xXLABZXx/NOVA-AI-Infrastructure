"""Checkpoint save/load with architecture compatibility checks.

Weights are treated as opaque bytes: this module never imports torch, so
checkpoint bookkeeping works on machines without a training stack. The
JSON sidecar records the architecture/tokenizer/vocabulary the weights
were produced with, and :meth:`CheckpointManager.load` refuses to hand
back a checkpoint that does not match the expected configuration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: Metadata keys that must match between a checkpoint and the expected config.
COMPATIBILITY_KEYS = ("architecture", "vocab_size", "tokenizer")


class IncompatibleCheckpointError(Exception):
    """Raised when a checkpoint does not match the expected configuration."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    """Save and load training checkpoints in ``checkpoint_dir``.

    Layout: ``<checkpoint_dir>/<tag>/checkpoint.json`` (+ optional
    ``weights.bin``).
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, section: dict) -> "CheckpointManager":
        """Build from a config section dict (uses ``checkpoint_dir``)."""
        return cls(section.get("checkpoint_dir", "checkpoints"))

    def _tag_dir(self, tag: str) -> Path:
        return self.checkpoint_dir / tag

    def save(
        self,
        tag: str,
        metadata: dict,
        weights: bytes | None = None,
    ) -> Path:
        """Save a checkpoint sidecar (and optional opaque weights).

        Returns the path of the written ``checkpoint.json``.
        """
        tag_dir = self._tag_dir(tag)
        tag_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": tag,
            "saved_at": _utcnow(),
            **dict(metadata),
        }
        meta_path = tag_dir / "checkpoint.json"
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if weights is not None:
            (tag_dir / "weights.bin").write_bytes(bytes(weights))
        return meta_path

    def load(self, tag: str, expected: dict) -> dict:
        """Load checkpoint metadata after verifying compatibility.

        ``expected`` must contain the :data:`COMPATIBILITY_KEYS`; any
        mismatch raises :class:`IncompatibleCheckpointError` with an
        explicit message. Returns
        ``{"metadata": ..., "weights_path": Path | None}``.
        """
        meta_path = self._tag_dir(tag) / "checkpoint.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"no checkpoint found for tag {tag!r}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in COMPATIBILITY_KEYS:
            want = expected.get(key)
            have = metadata.get(key)
            if have != want:
                raise IncompatibleCheckpointError(
                    f"incompatible checkpoint {tag!r}: {key} mismatch "
                    f"(expected {want!r}, found {have!r})"
                )
        weights_path = self._tag_dir(tag) / "weights.bin"
        return {
            "metadata": metadata,
            "weights_path": weights_path if weights_path.is_file() else None,
        }

    def latest_tag(self) -> str | None:
        """Return the tag of the most recently saved checkpoint, if any."""
        best: tuple[str, str] | None = None  # (saved_at, tag)
        for tag_dir in self.checkpoint_dir.iterdir():
            meta_path = tag_dir / "checkpoint.json"
            if not tag_dir.is_dir() or not meta_path.is_file():
                continue
            try:
                saved_at = json.loads(
                    meta_path.read_text(encoding="utf-8")
                ).get("saved_at", "")
            except (OSError, json.JSONDecodeError):
                continue
            if best is None or saved_at >= best[0]:
                best = (saved_at, tag_dir.name)
        return best[1] if best else None

    def resume_latest(self, expected: dict) -> dict | None:
        """Load the latest checkpoint, or return None when there is none."""
        tag = self.latest_tag()
        if tag is None:
            return None
        return self.load(tag, expected)
