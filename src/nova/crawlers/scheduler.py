"""Concurrent fetch scheduler for the NOVA crawlers.

A single scheduler thread owns a :class:`ThreadPoolExecutor` and dispatches
URL fetches from a :class:`~nova.crawlers.frontier.Frontier`. Politeness is
never sacrificed for speed:

- every host gets its own minimum inter-request delay (robots.txt
  ``crawl-delay`` when present, otherwise the configured default);
- an optional global token bucket caps the overall request rate;
- 429/5xx responses trigger host-level exponential backoff with jitter;
- per-host and global page budgets bound the crawl.

The actual fetching happens inside worker threads through the caller's
fetcher (the sandbox HTTP client), so every sandbox safety check
(SSRF guard, allow/blocklists, size caps) stays on the request path.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import threading
import time
from typing import Callable, Protocol

log = logging.getLogger(__name__)

__all__ = [
    "HostPoliteness",
    "backoff_delay",
    "TokenBucket",
    "CrawlScheduler",
    "FetchHandler",
]


class FetchHandler(Protocol):
    """What the scheduler needs from the crawler using it."""

    def scheduler_prepare(self, row: dict) -> tuple[bool, str]:
        """Decide whether a claimed row may be dispatched.

        Runs in the scheduler thread. Return ``(True, "")`` to dispatch or
        ``(False, reason)`` to skip the row (``reason`` in
        ``{"seen", "robots", "filtered", "depth", ...}``).
        """
        ...

    def scheduler_host_delay(self, host: str) -> float:
        """Minimum delay between two requests to ``host`` (seconds)."""
        ...

    def scheduler_process(self, row: dict) -> None:
        """Fetch the URL and handle success / retry / dead-letter.

        Runs in a worker thread. Must never raise: unexpected exceptions are
        routed to :meth:`scheduler_crash` by the scheduler.
        """
        ...

    def scheduler_crash(self, row: dict, exc: BaseException) -> None:
        """Last-resort handler for an unexpected worker exception."""
        ...


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 60.0,
                  jitter_ratio: float = 0.25) -> float:
    """Exponential backoff with jitter: ``base * 2**(attempt-1)`` capped.

    ``attempt`` is 1-based (1 = first retry). Jitter is uniform in
    ``[1 - jitter_ratio, 1 + jitter_ratio]`` so hosts don't retry in lockstep.
    """
    delay = min(cap, base * (2.0 ** max(0, attempt - 1)))
    jitter = 1.0 + random.uniform(-jitter_ratio, jitter_ratio)
    return max(0.0, delay * jitter)


class TokenBucket:
    """Minimal thread-safe token bucket (requests per minute)."""

    def __init__(self, rate_per_minute: float, capacity: int | None = None):
        self._rate = float(rate_per_minute) / 60.0
        self._capacity = float(capacity if capacity else rate_per_minute)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 5.0) -> bool:
        """Block up to ``timeout`` seconds for one token."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)


class HostPoliteness:
    """Per-host minimum inter-request delay tracking.

    ``delay_fn(host)`` resolves the delay for a host (robots crawl-delay or
    the configured default). All state is guarded by a lock; the scheduler
    thread is the only writer.
    """

    def __init__(self, default_delay: float,
                 delay_fn: Callable[[str], float] | None = None):
        self.default_delay = max(0.0, float(default_delay))
        self._delay_fn = delay_fn or (lambda host: self.default_delay)
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def delay_for(self, host: str) -> float:
        try:
            return max(0.0, float(self._delay_fn(host)))
        except Exception:
            return self.default_delay

    def is_ready(self, host: str, now: float) -> bool:
        with self._lock:
            return now >= self._next_allowed.get(host, 0.0)

    def note_dispatch(self, host: str, now: float | None = None,
                      delay: float | None = None) -> float:
        """Record a dispatch; return the delay applied for ``host``.

        Uses ``delay`` when given, otherwise resolves via ``delay_fn``.
        """
        now = time.monotonic() if now is None else now
        resolved = max(0.0, float(delay)) if delay is not None else self.delay_for(host)
        with self._lock:
            self._next_allowed[host] = now + resolved
        return resolved


class CrawlScheduler:
    """Dispatch frontier URLs to a thread pool with per-host politeness.

    Parameters
    ----------
    max_workers:
        Bounded global concurrency (worker threads).
    default_host_delay:
        Fallback minimum seconds between requests to the same host.
    requests_per_minute:
        Global token-bucket rate; ``0`` disables the bucket.
    max_pages_total:
        Stop claiming new URLs after this many dispatches (``0`` = unlimited).
    """

    def __init__(self, *, max_workers: int = 10,
                 default_host_delay: float = 1.0,
                 requests_per_minute: float = 0,
                 max_pages_total: int = 0):
        self.max_workers = max(1, int(max_workers))
        self.default_host_delay = max(0.0, float(default_host_delay))
        self.max_pages_total = max(0, int(max_pages_total))
        self._bucket = (TokenBucket(requests_per_minute)
                        if requests_per_minute > 0 else None)

    def run(self, frontier, handler: FetchHandler,
            politeness: HostPoliteness | None = None) -> dict:
        """Run the crawl until the frontier is drained. Never raises."""
        politeness = politeness or HostPoliteness(self.default_host_delay)
        stats = {"dispatched": 0, "completed": 0, "skipped": 0, "crashed": 0}
        recovered = frontier.reset_stale()
        if recovered:
            log.info("frontier: recovered %d abandoned in-progress URLs",
                     recovered)

        max_pages_per_host = getattr(handler, "max_pages_per_host", 0) or 0

        def host_ok(host: str) -> bool:
            now = time.monotonic()
            if not politeness.is_ready(host, now):
                return False
            try:
                if frontier.host_in_backoff(host, now):
                    return False
            except Exception:
                pass
            return True

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="nova-crawl"
        ) as pool:
            in_flight: dict[concurrent.futures.Future, dict] = {}
            dispatched = 0
            while True:
                budget_exhausted = (
                    self.max_pages_total and dispatched >= self.max_pages_total
                )
                # Reap finished workers.
                for future in [f for f in in_flight if f.done()]:
                    row = in_flight.pop(future)
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        log.debug("worker raised for %s: %s", row.get("url"), exc)
                        stats["crashed"] += 1
                    stats["completed"] += 1

                # Dispatch while workers are free.
                while len(in_flight) < self.max_workers:
                    if self.max_pages_total and dispatched >= self.max_pages_total:
                        break
                    now = time.monotonic()
                    row = frontier.claim_next_ready(
                        now, host_ok, max_pages_per_host=max_pages_per_host)
                    if row is None:
                        break
                    ok, reason = handler.scheduler_prepare(row)
                    if not ok:
                        frontier.mark_done(row["url"], status="skipped")
                        try:
                            frontier.log_attempt(row["url"], row["attempts"],
                                                 f"skipped:{reason}", "")
                        except Exception:
                            pass
                        stats["skipped"] += 1
                        continue
                    delay = handler.scheduler_host_delay(row["host"])
                    politeness.note_dispatch(row["host"], now, delay=delay)
                    if self._bucket is not None:
                        self._bucket.acquire(timeout=10.0)
                    future = pool.submit(self._guarded_process, handler, row)
                    in_flight[future] = row
                    dispatched += 1
                    stats["dispatched"] += 1

                if not in_flight:
                    if budget_exhausted or frontier.pending_count() == 0:
                        break
                    nxt = frontier.next_due_at()
                    now = time.monotonic()
                    if nxt is None or nxt <= now:
                        # Nothing claimable but work remains: hosts are cooling
                        # down (politeness/backoff). Wait briefly and re-check.
                        time.sleep(0.1)
                        # Avoid an infinite nap when nothing can ever proceed.
                        if frontier.pending_count() == 0:
                            break
                        # If still nothing claimable after a bounded wait,
                        # re-check rather than sleep forever.
                        continue
                    time.sleep(min(max(0.0, nxt - now), 1.0))
                else:
                    concurrent.futures.wait(
                        list(in_flight), timeout=0.25,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
        return stats

    @staticmethod
    def _guarded_process(handler: FetchHandler, row: dict) -> None:
        try:
            handler.scheduler_process(row)
        except Exception as exc:
            log.debug("scheduler_process raised for %s: %s", row.get("url"), exc)
            try:
                handler.scheduler_crash(row, exc)
            except Exception:
                pass
