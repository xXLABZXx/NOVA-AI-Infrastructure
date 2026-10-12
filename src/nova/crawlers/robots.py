"""robots.txt compliance via :mod:`urllib.robotparser`.

Per-host parsers are cached with a configurable TTL (default 24h); the cache
can be backed by the crawl frontier's SQLite ``robots_cache`` table so it
survives restarts. Also exposes ``crawl-delay`` and ``Sitemap:`` directives
found in robots.txt.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from nova.crawlers.discovery import discover_sitemaps

log = logging.getLogger(__name__)

__all__ = ["RobotsChecker"]


class _Entry:
    __slots__ = ("parser", "raw_body", "fetched_at")

    def __init__(self, parser: RobotFileParser, raw_body: str, fetched_at: float):
        self.parser = parser
        self.raw_body = raw_body
        self.fetched_at = fetched_at  # wall-clock seconds (time.time())


class RobotsChecker:
    """Check ``can_fetch`` against each site's ``/robots.txt``.

    The robots file is fetched through the provided ``fetcher`` (same protocol
    as the crawler uses) and parsers are cached per ``(scheme, netloc)`` with
    a TTL. Any failure to fetch or parse robots.txt fails open (fetching is
    allowed) and is logged at debug level.

    Parameters
    ----------
    ttl_hours:
        How long a cached robots.txt stays valid.
    cache:
        Optional persistent cache with ``get_robots(host) -> (body,
        fetched_at) | None``, ``set_robots(host, body, sitemaps)`` and
        ``get_robots_sitemaps(host)`` — satisfied by
        :class:`~nova.crawlers.frontier.Frontier`.
    """

    def __init__(self, user_agent: str, fetcher, *, ttl_hours: float = 24.0,
                 cache=None):
        self.user_agent = user_agent
        self.fetcher = fetcher
        self.ttl_hours = max(0.0, float(ttl_hours))
        self.cache = cache
        self._entries: dict[tuple[str, str], _Entry] = {}

    @classmethod
    def from_config(cls, section: dict, *, fetcher, cache=None) -> "RobotsChecker":
        """Build from a config section (uses ``user_agent`` / ``robots_ttl_hours``)."""
        return cls(
            user_agent=section.get("user_agent", "NOVA-ResearchBot/0.1"),
            fetcher=fetcher,
            ttl_hours=float(section.get("robots_ttl_hours", 24.0)),
            cache=cache,
        )

    # -- loading ------------------------------------------------------------
    def _load_entry(self, scheme: str, netloc: str) -> _Entry:
        robots_url = f"{scheme}://{netloc}/robots.txt"
        raw_body = ""
        try:
            result = self.fetcher.fetch(robots_url)
            body = result.body or b""
            if result.status_code == 200 and body:
                raw_body = body.decode("utf-8", errors="replace")
        except Exception as exc:  # fail open: a broken robots.txt must not block us
            log.debug("robots.txt fetch failed for %s: %s", robots_url, exc)

        parser = RobotFileParser()
        if raw_body:
            parser.parse(raw_body.splitlines())
        else:
            # No rules: a never-parsed RobotFileParser answers can_fetch()
            # with False (assumes unread), so set allow_all explicitly.
            parser.allow_all = True

        entry = _Entry(parser, raw_body, time.time())
        if self.cache is not None and raw_body:
            try:
                sitemaps = discover_sitemaps(raw_body, robots_url)
                self.cache.set_robots(netloc, raw_body, sitemaps)
            except Exception as exc:
                log.debug("robots cache write failed for %s: %s", netloc, exc)
        return entry

    def _entry_for(self, scheme: str, netloc: str) -> _Entry:
        key = (scheme, netloc)
        entry = self._entries.get(key)
        now = time.time()
        if entry is not None and (now - entry.fetched_at) < self.ttl_hours * 3600:
            return entry
        # Try the persistent cache before hitting the network.
        if self.cache is not None:
            try:
                cached = self.cache.get_robots(netloc)
            except Exception:
                cached = None
            if cached is not None:
                body, fetched_at = cached
                if (now - fetched_at) < self.ttl_hours * 3600 and body:
                    parser = RobotFileParser()
                    parser.parse(body.splitlines())
                    entry = _Entry(parser, body, fetched_at)
                    self._entries[key] = entry
                    return entry
        entry = self._load_entry(scheme, netloc)
        self._entries[key] = entry
        return entry

    # -- public API -----------------------------------------------------------
    def can_fetch(self, url: str) -> bool:
        """Return True when ``url`` is allowed for our user agent."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            entry = self._entry_for(parsed.scheme or "http", parsed.netloc)
            return bool(entry.parser.can_fetch(self.user_agent, url))
        except Exception as exc:
            log.debug("robots check failed for %s: %s (allowing)", url, exc)
            return True

    def crawl_delay(self, url: str) -> float | None:
        """The robots.txt ``crawl-delay`` for our user agent, if declared."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return None
            entry = self._entry_for(parsed.scheme or "http", parsed.netloc)
            delay = entry.parser.crawl_delay(self.user_agent)
            return float(delay) if delay is not None else None
        except Exception as exc:
            log.debug("crawl-delay lookup failed for %s: %s", url, exc)
            return None

    def sitemaps(self, url: str) -> list[str]:
        """``Sitemap:`` URLs declared in the host's robots.txt."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return []
            # Prefer the persistent cache (no extra parsing).
            if self.cache is not None:
                try:
                    cached = self.cache.get_robots_sitemaps(parsed.netloc)
                    if cached:
                        return list(cached)
                except Exception:
                    pass
            entry = self._entry_for(parsed.scheme or "http", parsed.netloc)
            robots_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/robots.txt"
            return discover_sitemaps(entry.raw_body, robots_url)
        except Exception as exc:
            log.debug("sitemap lookup failed for %s: %s", url, exc)
            return []

    def clear_cache(self) -> None:
        """Drop all cached robots.txt parsers."""
        self._entries.clear()
