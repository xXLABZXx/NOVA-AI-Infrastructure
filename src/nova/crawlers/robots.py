"""robots.txt compliance via :mod:`urllib.robotparser`."""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

log = logging.getLogger(__name__)

__all__ = ["RobotsChecker"]


class RobotsChecker:
    """Check ``can_fetch`` against each site's ``/robots.txt``.

    The robots file is fetched through the provided ``fetcher`` (same protocol
    as the crawler uses) and parsers are cached per ``(scheme, netloc)``. Any
    failure to fetch or parse robots.txt fails open (fetching is allowed) and
    is logged at debug level.
    """

    def __init__(self, user_agent: str, fetcher):
        self.user_agent = user_agent
        self.fetcher = fetcher
        self._parsers: dict[tuple[str, str], RobotFileParser] = {}

    @classmethod
    def from_config(cls, section: dict, *, fetcher) -> "RobotsChecker":
        """Build from a config section (uses its ``user_agent`` when present)."""
        return cls(user_agent=section.get("user_agent", "NOVA-ResearchBot/0.1"),
                   fetcher=fetcher)

    def _load_parser(self, scheme: str, netloc: str) -> RobotFileParser:
        parser = RobotFileParser()
        robots_url = f"{scheme}://{netloc}/robots.txt"
        try:
            result = self.fetcher.fetch(robots_url)
            body = result.body or b""
            if result.status_code == 200 and body:
                lines = body.decode("utf-8", errors="replace").splitlines()
                parser.parse(lines)
                return parser
            # Non-200 / empty body: no rules -> allow everything.
        except Exception as exc:  # fail open: a broken robots.txt must not block us
            log.debug("robots.txt fetch failed for %s: %s", robots_url, exc)
        # Mark explicitly as "no rules": a never-parsed RobotFileParser
        # answers can_fetch() with False (assumes unread), so set allow_all.
        parser.allow_all = True
        return parser

    def can_fetch(self, url: str) -> bool:
        """Return True when ``url`` is allowed for our user agent."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            key = (parsed.scheme or "http", parsed.netloc)
            parser = self._parsers.get(key)
            if parser is None:
                parser = self._load_parser(*key)
                self._parsers[key] = parser
            allowed = parser.can_fetch(self.user_agent, url)
            return bool(allowed)
        except Exception as exc:
            log.debug("robots check failed for %s: %s (allowing)", url, exc)
            return True

    def clear_cache(self) -> None:
        """Drop all cached robots.txt parsers."""
        self._parsers.clear()
