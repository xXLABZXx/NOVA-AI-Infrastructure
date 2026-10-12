"""Rate-limiting primitives (standard library only)."""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe token bucket refilled at ``rate_per_minute``.

    ``capacity`` defaults to one minute's worth of tokens. Tokens accrue
    continuously (fractional), using :func:`time.monotonic` so wall-clock
    changes cannot break the limiter.
    """

    def __init__(self, rate_per_minute: float, capacity: int | None = None):
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        if capacity is None:
            capacity = rate_per_minute
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rate_per_minute = float(rate_per_minute)
        self.capacity = float(capacity)
        self._rate_per_second = self.rate_per_minute / 60.0
        self._tokens = self.capacity  # bucket starts full
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Add tokens earned since the last refill. Caller must hold the lock."""
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(
                self.capacity, self._tokens + elapsed * self._rate_per_second
            )
            self._last = now

    def acquire(self, timeout: float = 0) -> bool:
        """Consume one token if one becomes available within ``timeout`` seconds.

        Returns True when a token was consumed, False when the wait timed out.
        ``timeout=0`` (the default) never waits.
        """
        timeout = max(0.0, timeout)
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # How long until the next token accrues.
                wait_for_token = (1.0 - self._tokens) / self._rate_per_second
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, wait_for_token, 0.05))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"TokenBucket(rate_per_minute={self.rate_per_minute}, "
            f"capacity={self.capacity})"
        )
