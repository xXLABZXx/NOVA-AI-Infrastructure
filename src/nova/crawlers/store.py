"""Persistent "already seen" URL store backed by sqlite3.

Lets crawls resume without refetching pages and supports conditional
revalidation via ETag / Last-Modified headers.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

__all__ = ["SeenStore"]


def _normalize_url(url: str) -> str:
    """Canonical form used for hashing: drop fragment, lowercase scheme/host."""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        return urlunparse((scheme, netloc, parsed.path or "/", parsed.params,
                           parsed.query, ""))
    except Exception:
        return url.strip()


def _url_hash(url: str) -> str:
    return hashlib.sha256(_normalize_url(url).encode("utf-8")).hexdigest()


class SeenStore:
    """SQLite-backed record of crawled URLs.

    Parameters
    ----------
    db_path:
        Path of the sqlite database file. Parent directories are created.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS seen (
                   url_hash TEXT PRIMARY KEY,
                   url TEXT NOT NULL,
                   etag TEXT,
                   last_modified TEXT,
                   seen_at TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "SeenStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- API --------------------------------------------------------------
    def has(self, url: str) -> bool:
        """Whether ``url`` was already crawled."""
        row = self._conn.execute(
            "SELECT 1 FROM seen WHERE url_hash = ?", (_url_hash(url),)
        ).fetchone()
        return row is not None

    def mark(self, url: str, etag: str | None = None,
             last_modified: str | None = None) -> None:
        """Record ``url`` as seen, optionally with revalidation headers."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO seen (url_hash, url, etag, last_modified, seen_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(url_hash) DO UPDATE SET
                   etag=excluded.etag,
                   last_modified=excluded.last_modified,
                   seen_at=excluded.seen_at""",
            (_url_hash(url), _normalize_url(url), etag, last_modified, now),
        )
        self._conn.commit()

    def conditional_headers(self, url: str) -> dict:
        """Return ``If-None-Match``/``If-Modified-Since`` headers for ``url``.

        Empty dict when the URL was never seen or no validators were stored.
        """
        row = self._conn.execute(
            "SELECT etag, last_modified FROM seen WHERE url_hash = ?",
            (_url_hash(url),),
        ).fetchone()
        headers: dict[str, str] = {}
        if row:
            etag, last_modified = row
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified
        return headers

    def count(self) -> int:
        """Number of URLs recorded."""
        row = self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        try:
            self._conn.commit()
        finally:
            self._conn.close()
