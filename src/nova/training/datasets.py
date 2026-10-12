"""Dataset split management and held-out leak prevention.

Splits are assigned deterministically from the document id (sha256 mod
100), so the same corpus always yields the same split assignment without
storing any state. :class:`SplitEnforcer` additionally guards the
training data path: any record carrying a held-out split is rejected,
and :meth:`SplitEnforcer.assert_no_leak` catches near-duplicate text
leaking from training into evaluation.
"""

from __future__ import annotations

import hashlib
import re

#: The valid dataset splits.
SPLITS = ("train", "val", "test", "challenge")

#: Splits that must never be fed to the trainer.
HELD_OUT_SPLITS = ("val", "test", "challenge")

_DEFAULT_RATIOS = {"train": 80, "val": 10, "test": 5, "challenge": 5}


class HeldOutLeakError(Exception):
    """Raised when held-out data would leak into training."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class SplitEnforcer:
    """Deterministic split assignment + leak guards.

    ``split_map`` optionally pins specific doc ids to a split (overrides
    the hash assignment). ``ratios`` optionally replaces the default
    80/10/5/5 split percentages; values must sum to 100.
    """

    def __init__(
        self,
        split_map: dict[str, str] | None = None,
        ratios: dict[str, int] | None = None,
    ) -> None:
        self.split_map = dict(split_map or {})
        for doc_id, split in self.split_map.items():
            if split not in SPLITS:
                raise ValueError(
                    f"split_map[{doc_id!r}] = {split!r} is not a valid split"
                )
        self.ratios = dict(ratios or _DEFAULT_RATIOS)
        if set(self.ratios) - set(SPLITS):
            raise ValueError(f"ratios keys must be a subset of {SPLITS}")
        if sum(self.ratios.values()) != 100:
            raise ValueError(
                f"ratios must sum to 100, got {sum(self.ratios.values())}"
            )
        # Cumulative thresholds over the canonical split order.
        self._thresholds: list[tuple[int, str]] = []
        cumulative = 0
        for split in SPLITS:
            cumulative += self.ratios.get(split, 0)
            self._thresholds.append((cumulative, split))

    @classmethod
    def from_config(cls, section: dict) -> "SplitEnforcer":
        """Build from a config section dict (uses ``ratios`` / ``split_map``)."""
        return cls(
            split_map=section.get("split_map"),
            ratios=section.get("ratios"),
        )

    def assign(self, doc_id: str) -> str:
        """Deterministically assign ``doc_id`` to a split."""
        if doc_id in self.split_map:
            return self.split_map[doc_id]
        bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 100
        for threshold, split in self._thresholds:
            if bucket < threshold:
                return split
        return "train"  # unreachable when ratios sum to 100, kept for safety

    def load_for_training(self, records: list[dict]) -> list[dict]:
        """Return the trainable records, rejecting any held-out split.

        Records without a ``"split"`` field are treated as ``"train"``.
        Raises :class:`HeldOutLeakError` if any record carries a held-out
        split (val/test/challenge).
        """
        offenders = [
            str(r.get("id", "?"))
            for r in records
            if r.get("split") in HELD_OUT_SPLITS
        ]
        if offenders:
            preview = ", ".join(offenders[:5])
            raise HeldOutLeakError(
                f"{len(offenders)} held-out record(s) would leak into "
                f"training (e.g. {preview})"
            )
        return [r for r in records if r.get("split", "train") == "train"]

    def assert_no_leak(
        self, train_texts: list[str], heldout_texts: list[str]
    ) -> None:
        """Raise :class:`HeldOutLeakError` if normalized texts overlap.

        Normalization strips, lowercases, and collapses whitespace, so
        trivial reformatting does not hide a leak. The error reports how
        many held-out texts collide (plus a few samples).
        """
        train_hashes = {
            hashlib.sha256(_normalize(t).encode("utf-8")).hexdigest()
            for t in train_texts
        }
        collisions = [
            t for t in heldout_texts
            if hashlib.sha256(_normalize(t).encode("utf-8")).hexdigest()
            in train_hashes
        ]
        if collisions:
            sample = "; ".join(repr(c[:60]) for c in collisions[:3])
            raise HeldOutLeakError(
                f"{len(collisions)} held-out text(s) also appear in training "
                f"data (e.g. {sample})"
            )


def check_dataset_compatibility(dataset_info: dict, expected: dict) -> None:
    """Verify a dataset matches the expected tokenizer/vocab/format.

    Raises ValueError with a clear message on the first mismatch.
    """
    for key in ("tokenizer", "vocab_size", "format"):
        want = expected.get(key)
        have = dataset_info.get(key)
        if have != want:
            raise ValueError(
                f"dataset incompatible: {key} mismatch "
                f"(expected {want!r}, found {have!r})"
            )
