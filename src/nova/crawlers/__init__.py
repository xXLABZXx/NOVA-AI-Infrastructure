"""NOVA web crawlers: concurrent polite crawling with a persistent frontier.

Main entry points: :class:`GeneralCrawler`, :class:`DocsCrawler`,
:class:`CodeCrawler`, :class:`PapersCrawler`, plus the :class:`SeenStore`
(persistent URL ledger), :class:`Frontier` (SQLite work queue),
:class:`CrawlScheduler` (thread-pool dispatcher), :class:`RobotsChecker`,
URL canonicalization, sitemap/feed discovery and HTML extraction helpers.
"""

from nova.crawlers.base import BaseCrawler, CrawlResult
from nova.crawlers.canonical import canonicalize_url, url_fingerprint
from nova.crawlers.code import CodeCrawler
from nova.crawlers.discovery import (
    discover_feeds,
    discover_sitemaps,
    is_sitemap_index,
    iter_sitemap_urls,
    parse_feed,
    parse_sitemap,
    parse_sitemap_detailed,
)
from nova.crawlers.docs import DocsCrawler
from nova.crawlers.extract import (
    detect_language,
    detect_language_with_confidence,
    extract_article,
    extract_code_text,
    extract_paper_metadata,
    extract_text,
    score_quality,
)
from nova.crawlers.frontier import Frontier, compute_priority
from nova.crawlers.general import GeneralCrawler
from nova.crawlers.papers import PapersCrawler
from nova.crawlers.robots import RobotsChecker
from nova.crawlers.scheduler import CrawlScheduler, HostPoliteness, backoff_delay
from nova.crawlers.store import SeenStore

__all__ = [
    "BaseCrawler",
    "GeneralCrawler",
    "DocsCrawler",
    "CodeCrawler",
    "PapersCrawler",
    "CrawlResult",
    "SeenStore",
    "Frontier",
    "CrawlScheduler",
    "HostPoliteness",
    "RobotsChecker",
    "backoff_delay",
    "compute_priority",
    "canonicalize_url",
    "url_fingerprint",
    "extract_text",
    "extract_code_text",
    "extract_article",
    "extract_paper_metadata",
    "detect_language",
    "detect_language_with_confidence",
    "score_quality",
    "discover_sitemaps",
    "parse_sitemap",
    "parse_sitemap_detailed",
    "is_sitemap_index",
    "iter_sitemap_urls",
    "discover_feeds",
    "parse_feed",
]
