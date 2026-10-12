"""General-purpose crawler: follows all http(s) links."""

from __future__ import annotations

from nova.crawlers.base import BaseCrawler

__all__ = ["GeneralCrawler"]


class GeneralCrawler(BaseCrawler):
    """Breadth-first crawl over all http(s) links.

    ``should_follow`` accepts every http(s) URL; no domain restriction beyond
    the base hook. Inherits ``from_config`` from :class:`BaseCrawler`.
    """
