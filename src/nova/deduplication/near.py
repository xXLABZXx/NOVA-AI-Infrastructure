"""Near-duplicate detection with SimHash (stdlib only).

SimHash maps a document to a short bit vector such that near-duplicate
documents get fingerprints with a small Hamming distance. Shingles are
word 5-grams hashed with BLAKE2b; shingle frequencies weight the vote.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

__all__ = ["SimHash", "NearDeduper"]

_SHINGLE_SIZE = 5


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class SimHash:
    """64-bit (default) SimHash fingerprint of a text."""

    def __init__(self, text: str, hashbits: int = 64):
        if hashbits <= 0 or hashbits > 256:
            raise ValueError("hashbits must be between 1 and 256")
        self.hashbits = hashbits
        self.fingerprint: int = self._compute(text or "", hashbits)

    @classmethod
    def _compute(cls, text: str, hashbits: int) -> int:
        tokens = _tokens(text)
        if len(tokens) >= _SHINGLE_SIZE:
            shingles = [
                " ".join(tokens[i:i + _SHINGLE_SIZE])
                for i in range(len(tokens) - _SHINGLE_SIZE + 1)
            ]
        elif tokens:
            shingles = [" ".join(tokens)]
        else:
            return 0
        weights = Counter(shingles)

        vector = [0] * hashbits
        digest_size = max(1, min(64, (hashbits + 7) // 8))
        for shingle, weight in weights.items():
            digest = hashlib.blake2b(
                shingle.encode("utf-8"), digest_size=digest_size).digest()
            hashed = int.from_bytes(digest, "big")
            for i in range(hashbits):
                if (hashed >> i) & 1:
                    vector[i] += weight
                else:
                    vector[i] -= weight

        fingerprint = 0
        for i, value in enumerate(vector):
            if value > 0:
                fingerprint |= 1 << i
        return fingerprint

    def hamming(self, other: "SimHash") -> int:
        """Hamming distance between two fingerprints (must share hashbits)."""
        if self.hashbits != other.hashbits:
            raise ValueError("cannot compare SimHashes with different hashbits")
        return bin(self.fingerprint ^ other.fingerprint).count("1")

    def similarity(self, other: "SimHash") -> float:
        """Similarity in [0, 1]: 1 - hamming_distance / hashbits."""
        return 1.0 - self.hamming(other) / self.hashbits

    def __eq__(self, other) -> bool:
        return (isinstance(other, SimHash)
                and self.hashbits == other.hashbits
                and self.fingerprint == other.fingerprint)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"SimHash(fingerprint={self.fingerprint:#x}, hashbits={self.hashbits})"


class NearDeduper:
    """Flags documents whose SimHash is within ``threshold_bits`` of a seen one."""

    def __init__(self, threshold_bits: int = 8, hashbits: int = 64):
        if threshold_bits < 0:
            raise ValueError("threshold_bits must be non-negative")
        self.threshold_bits = threshold_bits
        self.hashbits = hashbits
        self._fingerprints: list[SimHash] = []

    @classmethod
    def from_config(cls, section: dict) -> "NearDeduper":
        """Build from a config section (``threshold_bits`` / ``hashbits``)."""
        return cls(
            threshold_bits=int(section.get("threshold_bits", 8)),
            hashbits=int(section.get("hashbits", 64)),
        )

    def _simhash(self, text: str) -> SimHash:
        return SimHash(text, hashbits=self.hashbits)

    def is_near_duplicate(self, text: str) -> bool:
        """Whether ``text`` is near-duplicate of anything already added."""
        candidate = self._simhash(text)
        return any(candidate.hamming(seen) <= self.threshold_bits
                   for seen in self._fingerprints)

    def add(self, text: str) -> None:
        """Record ``text``'s fingerprint as seen."""
        self._fingerprints.append(self._simhash(text))

    def __len__(self) -> int:
        return len(self._fingerprints)

    def dedupe(self, docs: list[dict], text_key: str = "text"):
        """Split ``docs`` into ``(unique, duplicates)``; first occurrence wins."""
        unique: list[dict] = []
        duplicates: list[dict] = []
        for doc in docs:
            text = doc.get(text_key, "")
            if self.is_near_duplicate(text):
                duplicates.append(doc)
            else:
                self.add(text)
                unique.append(doc)
        return unique, duplicates
