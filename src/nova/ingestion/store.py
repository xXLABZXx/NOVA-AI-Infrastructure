"""Append-only raw document store (JSONL)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterator

__all__ = ["RawStore"]


class RawStore:
    """Append-only JSONL store for raw crawled documents.

    There are deliberately no update or delete methods: raw evidence is
    immutable. Cleaning/dedup stages read from here and write elsewhere.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file so ``count()`` works on a fresh store.
        self.path.touch(exist_ok=True)

    def append(self, doc: dict) -> str:
        """Append ``doc`` as one JSON line; assign ``id`` when missing.

        The caller's dict is never mutated. Returns the document id.
        """
        record = dict(doc)
        doc_id = record.get("id") or uuid.uuid4().hex
        record["id"] = doc_id
        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return doc_id

    def iter_docs(self) -> Iterator[dict]:
        """Yield documents in file order (skips blank lines)."""
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load_all(self) -> list[dict]:
        """Load every document into a list."""
        return list(self.iter_docs())

    def count(self) -> int:
        """Number of stored documents."""
        return sum(1 for _ in self.iter_docs())
