"""SQLite-backed crawl frontier.

The frontier is the persistent work queue of the crawler: every URL to fetch,
its priority, retry state and outcome live here, so an interrupted crawl can
resume without refetching completed pages. It also hosts the robots.txt cache
and per-host crawl state (budgets, backoff).

All methods are safe to call from the scheduler's worker threads: a single
re-entrant lock serializes access to one shared connection
(``check_same_thread=False``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from nova.crawlers.canonical import canonicalize_url, url_fingerprint

__all__ = ["Frontier", "compute_priority"]

#: Row statuses stored in ``frontier.status``.
STATUS_QUEUED = "queued"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_DEAD = "dead"
STATUS_SKIPPED = "skipped"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    url            TEXT PRIMARY KEY,
    url_hash       TEXT NOT NULL,
    host           TEXT NOT NULL,
    priority       REAL NOT NULL DEFAULT 0,
    next_fetch_at  REAL NOT NULL DEFAULT 0,
    attempts       INTEGER NOT NULL DEFAULT 0,
    max_attempts   INTEGER NOT NULL DEFAULT 3,
    depth          INTEGER NOT NULL DEFAULT 0,
    discovered_from TEXT,
    kind           TEXT NOT NULL DEFAULT 'page',
    slevel         INTEGER NOT NULL DEFAULT 0,
    revalidate     INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'queued',
    last_error     TEXT,
    lastmod        TEXT,
    added_at       REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frontier_status ON frontier(status, next_fetch_at);
CREATE INDEX IF NOT EXISTS idx_frontier_host ON frontier(host, status);

CREATE TABLE IF NOT EXISTS robots_cache (
    host       TEXT PRIMARY KEY,
    body       TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    sitemaps   TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS host_state (
    host               TEXT PRIMARY KEY,
    pages_fetched      INTEGER NOT NULL DEFAULT 0,
    backoff_until      REAL NOT NULL DEFAULT 0,
    consecutive_errors INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS attempt_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    url        TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    at         REAL NOT NULL,
    outcome    TEXT NOT NULL,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempt_log_url ON attempt_log(url);
"""


def compute_priority(depth: int, seed_index: int | None = None,
                     boost: float = 0.0) -> float:
    """Priority score: shallower first, earlier seeds first, plus ``boost``.

    Higher values are claimed first (``ORDER BY priority DESC``).
    """
    score = 10.0 / (1.0 + max(0, depth))
    if seed_index is not None:
        score += 5.0 - min(seed_index, 200) * 0.01
    return score + boost


class Frontier:
    """Persistent, thread-safe URL frontier backed by SQLite.

    Parameters
    ----------
    db_path:
        Path of the sqlite file. ``":memory:"`` (default) keeps the frontier
        ephemeral — handy for unit tests and one-shot crawls.
    stale_timeout_seconds:
        ``in_progress`` rows older than this are treated as abandoned (e.g.
        after a crash) and re-queued by :meth:`reset_stale`.
    """

    def __init__(self, db_path: str | Path = ":memory:",
                 *, stale_timeout_seconds: float = 600.0):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Forward-compatible migration for DBs created before `revalidate`.
            try:
                self._conn.execute(
                    "ALTER TABLE frontier ADD COLUMN revalidate INTEGER "
                    "NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # column already exists
            self._conn.commit()

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    def __enter__(self) -> "Frontier":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- enqueue / claim --------------------------------------------------
    def add(self, url: str, *, depth: int = 0,
            discovered_from: str | None = None,
            priority: float | None = None, kind: str = "page",
            slevel: int = 0, max_attempts: int = 3,
            lastmod: str | None = None, revalidate: bool = False) -> bool:
        """Enqueue ``url``. Returns False when it was already present.

        The URL is canonicalized first, so tracking-parameter variants of an
        already-queued page are dropped here. ``revalidate`` marks the row
        for conditional revalidation even when the seen store knows the URL.
        """
        canon = canonicalize_url(url)
        if not canon:
            return False
        host = urlparse(canon).netloc
        if priority is None:
            priority = compute_priority(depth)
        now = time.monotonic()
        with self._lock:
            cursor = self._conn.execute(
                """INSERT OR IGNORE INTO frontier
                   (url, url_hash, host, priority, next_fetch_at, attempts,
                    max_attempts, depth, discovered_from, kind, slevel,
                    revalidate, status, lastmod, added_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)""",
                (canon, url_fingerprint(canon), host, priority, 0.0,
                 max_attempts, depth, discovered_from, kind, slevel,
                 1 if revalidate else 0, lastmod, now, now),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def claim_next_ready(self, now: float, host_ok, *,
                         max_pages_per_host: int = 0) -> dict | None:
        """Atomically claim the highest-priority due URL whose host is ready.

        ``host_ok(host)`` is evaluated in priority order; the first row whose
        host passes is marked ``in_progress`` and returned as a dict.
        Hosts over their page budget (``max_pages_per_host`` > 0) are skipped.
        Returns None when nothing is claimable right now.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT url, url_hash, host, priority, next_fetch_at,
                          attempts, max_attempts, depth, discovered_from,
                          kind, slevel, revalidate, status, last_error, lastmod
                   FROM frontier
                   WHERE status = 'queued' AND next_fetch_at <= ?
                   ORDER BY priority DESC, rowid ASC
                   LIMIT 40""",
                (now,),
            ).fetchall()
            for row in rows:
                host = row[2]
                if max_pages_per_host > 0 and self._host_pages(host) >= max_pages_per_host:
                    continue
                if not host_ok(host):
                    continue
                self._conn.execute(
                    "UPDATE frontier SET status='in_progress', updated_at=? "
                    "WHERE url=?",
                    (now, row[0]),
                )
                self._conn.commit()
                keys = ("url", "url_hash", "host", "priority", "next_fetch_at",
                        "attempts", "max_attempts", "depth", "discovered_from",
                        "kind", "slevel", "revalidate", "status", "last_error",
                        "lastmod")
                return dict(zip(keys, row))
            return None

    def next_due_at(self) -> float | None:
        """Earliest ``next_fetch_at`` among queued rows, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(next_fetch_at) FROM frontier WHERE status='queued'"
            ).fetchone()
            return row[0] if row and row[0] is not None else None

    def pending_count(self) -> int:
        """Rows still to process (queued or in_progress)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM frontier WHERE status IN ('queued','in_progress')"
            ).fetchone()
            return int(row[0]) if row else 0

    def reset_stale(self, now: float | None = None) -> int:
        """Re-queue ``in_progress`` rows abandoned longer than the timeout.

        Returns the number of rows recovered (crash recovery on startup).
        """
        now = time.monotonic() if now is None else now
        cutoff = now - self.stale_timeout_seconds
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE frontier SET status='queued', updated_at=? "
                "WHERE status='in_progress' AND updated_at < ?",
                (now, cutoff),
            )
            self._conn.commit()
            return cursor.rowcount

    # -- outcome recording --------------------------------------------------
    def mark_done(self, url: str, *, status: str = STATUS_DONE) -> None:
        now = time.monotonic()
        with self._lock:
            self._conn.execute(
                "UPDATE frontier SET status=?, updated_at=? WHERE url=?",
                (status, now, url),
            )
            self._conn.commit()

    def mark_retry(self, url: str, *, next_fetch_at: float, error: str,
                   attempts: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._conn.execute(
                """UPDATE frontier SET status='queued', next_fetch_at=?,
                       last_error=?, attempts=?, updated_at=? WHERE url=?""",
                (next_fetch_at, error, attempts, now, url),
            )
            self._conn.commit()

    def mark_dead(self, url: str, *, reason: str, error: str,
                  attempts: int | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            if attempts is None:
                self._conn.execute(
                    """UPDATE frontier SET status='dead', last_error=?,
                           updated_at=? WHERE url=?""",
                    (f"{reason}: {error}", now, url),
                )
            else:
                self._conn.execute(
                    """UPDATE frontier SET status='dead', last_error=?,
                           attempts=?, updated_at=? WHERE url=?""",
                    (f"{reason}: {error}", attempts, now, url),
                )
            self._conn.commit()

    def log_attempt(self, url: str, attempt_no: int, outcome: str,
                    detail: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO attempt_log (url, attempt_no, at, outcome, detail)"
                " VALUES (?, ?, ?, ?, ?)",
                (url, attempt_no, time.monotonic(), outcome, detail[:2000]),
            )
            self._conn.commit()

    def attempt_history(self, url: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT attempt_no, at, outcome, detail FROM attempt_log"
                " WHERE url=? ORDER BY attempt_no",
                (url,),
            ).fetchall()
            return [{"attempt_no": r[0], "at": r[1], "outcome": r[2],
                     "detail": r[3]} for r in rows]

    def dead_entries(self) -> list[dict]:
        """Permanently failed URLs with reason codes (for the dead-letter log)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT url, host, attempts, max_attempts, last_error, updated_at"
                " FROM frontier WHERE status='dead' ORDER BY updated_at"
            ).fetchall()
            return [{"url": r[0], "host": r[1], "attempts": r[2],
                     "max_attempts": r[3], "reason": r[4],
                     "dead_at_monotonic": r[5]} for r in rows]

    # -- host state ---------------------------------------------------------
    def _host_pages(self, host: str) -> int:
        row = self._conn.execute(
            "SELECT pages_fetched FROM host_state WHERE host=?", (host,)
        ).fetchone()
        return int(row[0]) if row else 0

    def host_pages(self, host: str) -> int:
        with self._lock:
            return self._host_pages(host)

    def inc_host_pages(self, host: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO host_state (host, pages_fetched)
                   VALUES (?, 1)
                   ON CONFLICT(host) DO UPDATE SET
                       pages_fetched = pages_fetched + 1""",
                (host,),
            )
            self._conn.commit()

    def host_in_backoff(self, host: str, now: float) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT backoff_until FROM host_state WHERE host=?", (host,)
            ).fetchone()
            return bool(row and row[0] > now)

    def set_host_backoff(self, host: str, until: float) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO host_state (host, backoff_until)
                   VALUES (?, ?)
                   ON CONFLICT(host) DO UPDATE SET
                       backoff_until = MAX(backoff_until, excluded.backoff_until)""",
                (host, until),
            )
            self._conn.commit()

    # -- robots cache ---------------------------------------------------------
    def get_robots(self, host: str) -> tuple[str, float] | None:
        """Return ``(body, fetched_at_wallclock)`` or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT body, fetched_at, sitemaps FROM robots_cache WHERE host=?",
                (host,),
            ).fetchone()
            if not row:
                return None
            return row[0], row[1]

    def get_robots_sitemaps(self, host: str) -> list[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT sitemaps FROM robots_cache WHERE host=?", (host,)
            ).fetchone()
            if not row:
                return []
            try:
                return list(json.loads(row[0] or "[]"))
            except Exception:
                return []

    def set_robots(self, host: str, body: str, sitemaps: list[str]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO robots_cache (host, body, fetched_at, sitemaps)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(host) DO UPDATE SET
                       body=excluded.body, fetched_at=excluded.fetched_at,
                       sitemaps=excluded.sitemaps""",
                (host, body, time.time(), json.dumps(list(sitemaps))),
            )
            self._conn.commit()
