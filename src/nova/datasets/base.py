"""Streaming primitives for the dataset factory.

Everything here is generator-based: the 17.4M-record base dataset is never
loaded into RAM. Parts may live inside a ``.zip`` archive (as shipped) or as
plain files; :func:`iter_jsonl_parts` handles both.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def _normalize(text: str) -> str:
    """Collapse whitespace + lowercase for dedup fingerprints."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def fingerprint(record: dict) -> str:
    """SHA-256 over normalized content for exact dedup.

    SFT records fingerprint on prompt+answer; text records on text.
    """
    if record.get("kind") == "text" or ("text" in record and "prompt" not in record):
        content = record.get("text", "")
    else:
        content = f"{record.get('prompt', '')}\n{record.get('answer', '')}"
    return hashlib.sha256(_normalize(content).encode("utf-8")).hexdigest()


@dataclass
class Record:
    """Canonical in-memory record (one JSONL line)."""

    id: str
    kind: str = "sft"  # "sft" | "text"
    category: str = "general"
    prompt: str = ""
    answer: str = ""
    text: str = ""
    source: str = ""
    synthetic: bool = False
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "category": self.category,
            "source": self.source,
            "synthetic": self.synthetic,
            "provenance": self.provenance,
        }
        if self.kind == "text":
            d["text"] = self.text
        else:
            d["prompt"] = self.prompt
            d["answer"] = self.answer
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        kind = d.get("kind", "sft")
        if kind not in ("sft", "text") and "text" in d and "prompt" not in d:
            kind = "text"
        return cls(
            id=str(d.get("id", "")),
            kind=kind,
            category=str(d.get("category", "general")),
            prompt=str(d.get("prompt", "")),
            answer=str(d.get("answer", "")),
            text=str(d.get("text", "")),
            source=str(d.get("source", "")),
            synthetic=bool(d.get("synthetic", False)),
            provenance=dict(d.get("provenance") or {}),
        )


class PartStats:
    """Counters for a streaming read."""

    def __init__(self) -> None:
        self.records = 0
        self.malformed = 0
        self.files = 0


def _iter_zip_lines(zip_path: str, member: str) -> Iterator[str]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for raw in fh:
                yield raw.decode("utf-8", errors="replace")


def iter_jsonl_parts(
    pattern: str,
    *,
    stats: PartStats | None = None,
    from_dict: bool = True,
) -> Iterator[dict]:
    """Yield parsed dicts from JSONL parts matching ``pattern``.

    ``pattern`` may point at plain files (glob) or at ``archive.zip`` /
    ``archive.zip:member-glob`` for zipped parts. Malformed lines are
    counted in ``stats.malformed`` and skipped — the stream never crashes.
    """
    stats = stats or PartStats()
    # zipped form: "path/to.zip" or "path/to.zip:member-*.txt"
    if ".zip" in pattern:
        zip_part, _, member_glob = pattern.partition(".zip")
        zip_path = zip_part + ".zip"
        member_glob = member_glob.lstrip(":") or "*"
        with zipfile.ZipFile(zip_path) as zf:
            members = sorted(
                n for n in zf.namelist()
                if glob.fnmatch.fnmatch(n, member_glob)
            )
        for member in members:
            stats.files += 1
            for line in _iter_zip_lines(zip_path, member):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    stats.malformed += 1
                    continue
                stats.records += 1
                yield rec
        return
    for path in sorted(glob.glob(pattern)):
        stats.files += 1
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    stats.malformed += 1
                    continue
                stats.records += 1
                yield rec


def iter_records(pattern: str, stats: PartStats | None = None) -> Iterator[Record]:
    """Same as :func:`iter_jsonl_parts` but yields :class:`Record` objects."""
    for d in iter_jsonl_parts(pattern, stats=stats):
        try:
            yield Record.from_dict(d)
        except Exception:
            if stats is not None:
                stats.malformed += 1


def write_jsonl(path: str | Path, records: Iterator[dict], *, append: bool = False) -> int:
    """Write dicts as JSONL (one per line). Returns the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count
