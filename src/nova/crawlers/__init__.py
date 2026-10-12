"""NOVA web crawlers: polite BFS crawling with robots.txt support.

Main entry points: :class:`GeneralCrawler`, :class:`DocsCrawler`,
:class:`CodeCrawler`, :class:`PapersCrawler`, plus the :class:`SeenStore`
(persistent URL ledger), :class:`RobotsChecker`, sitemap/feed discovery and
HTML text extraction helpers.
"""

from nova.crawlers.base import BaseCrawler, CrawlResult
from nova.crawlers.code import CodeCrawler
from nova.crawlers.discovery import discover_feeds, discover_sitemaps, parse_sitemap
from nova.crawlers.docs import DocsCrawler
from nova.crawlers.extract import (
    detect_language,
    extract_code_text,
    extract_paper_metadata,
    extract_text,
)
from nova.crawlers.general import GeneralCrawler
from nova.crawlers.papers import PapersCrawler
from nova.crawlers.robots import RobotsChecker
from nova.crawlers.store import SeenStore

__all__ = [
    "BaseCrawler",
    "GeneralCrawler",
    "DocsCrawler",
    "CodeCrawler",
    "PapersCrawler",
    "CrawlResult",
    "SeenStore",
    "RobotsChecker",
    "extract_text",
    "extract_code_text",
    "extract_paper_metadata",
    "detect_language",
    "discover_sitemaps",
    "parse_sitemap",
    "discover_feeds",
]
