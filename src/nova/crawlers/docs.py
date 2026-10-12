"""Documentation-focused crawler: prefers /docs/, /guide/, ... paths."""

from __future__ import annotations

from urllib.parse import urlparse

from nova.crawlers.base import BaseCrawler

__all__ = ["DocsCrawler"]

#: Path markers that strongly suggest documentation content.
_DOC_PATH_MARKERS = (
    "/docs/",
    "/documentation/",
    "/manual/",
    "/guide/",
    "/guides/",
    "/reference/",
    "/api/",
    "/handbook/",
)


class DocsCrawler(BaseCrawler):
    """Crawler that stays on documentation pages.

    A discovered link is followed when its path contains a documentation
    marker (``/docs/``, ``/guide/``, ...) or its host is listed in
    ``docs_allowlist`` (e.g. ``["docs.python.org"]``). Seeds are always
    fetched regardless of these rules.
    """

    def __init__(self, *, docs_allowlist: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.docs_allowlist = [h.lower() for h in (docs_allowlist or [])]

    @classmethod
    def from_config(cls, section: dict, *, fetcher, seen_store=None,
                    frontier=None) -> "DocsCrawler":
        """Build from a config section; reads ``docs_allowlist`` when present."""
        crawler = super().from_config(section, fetcher=fetcher,
                                      seen_store=seen_store, frontier=frontier)
        crawler.docs_allowlist = [
            h.lower() for h in section.get("docs_allowlist", []) or []
        ]
        return crawler

    def should_follow(self, url: str) -> bool:
        if not super().should_follow(url):
            return False
        parsed = urlparse(url)
        path = parsed.path.lower()
        if any(marker in path for marker in _DOC_PATH_MARKERS):
            return True
        return parsed.netloc.lower() in self.docs_allowlist
