"""Exact deduplication via SHA-256 fingerprints of normalized text."""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = ["ExactDeduper"]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ExactDeduper:
    """Remembers fingerprints of seen texts; flags byte-identical (modulo
    normalization) documents as duplicates."""

    def __init__(self):
        self._seen: set[str] = set()

    @classmethod
    def from_config(cls, section: dict) -> "ExactDeduper":
        """Build from a config section (no options currently; accepts any)."""
        return cls()

    @staticmethod
    def fingerprint(text: str) -> str:
        """SHA-256 hex of the normalized text."""
        return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Whether ``text`` was already added."""
        return self.fingerprint(text) in self._seen

    def add(self, text: str) -> None:
        """Record ``text`` as seen."""
        self._seen.add(self.fingerprint(text))

    def __len__(self) -> int:
        return len(self._seen)

    def dedupe(self, docs: list[dict], text_key: str = "text"):
        """Split ``docs`` into ``(unique, duplicates)``; first occurrence wins."""
        unique: list[dict] = []
        duplicates: list[dict] = []
        for doc in docs:
            text = doc.get(text_key, "")
            if self.is_duplicate(text):
                duplicates.append(doc)
            else:
                self.add(text)
                unique.append(doc)
        return unique, duplicates
