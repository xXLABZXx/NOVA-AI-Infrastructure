"""Sitemap and feed discovery helpers (stdlib only)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

__all__ = ["discover_sitemaps", "parse_sitemap", "discover_feeds"]

_SITEMAP_LINE = re.compile(r"(?im)^\s*Sitemap:\s*(\S+)\s*$")
_LINK_TAG = re.compile(r"<link\b[^>]*>", re.IGNORECASE)
_REL_ALTERNATE = re.compile(r"""\brel\s*=\s*["']alternate["']""", re.IGNORECASE)
_FEED_TYPE = re.compile(
    r"""\btype\s*=\s*["']application/(rss|atom)\+xml["']""", re.IGNORECASE
)
_HREF = re.compile(r"""\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def discover_sitemaps(robots_txt: str, base_url: str) -> list[str]:
    """Return sitemap URLs declared via ``Sitemap:`` lines in robots.txt.

    Relative references are resolved against ``base_url``.
    """
    if not robots_txt:
        return []
    urls = []
    for match in _SITEMAP_LINE.finditer(robots_txt):
        ref = match.group(1).strip()
        urls.append(urljoin(base_url, ref))
    # De-duplicate while preserving order.
    return list(dict.fromkeys(urls))


def parse_sitemap(xml_bytes: bytes | str) -> list[str]:
    """Return every ``<loc>`` URL in a sitemap (or sitemap index) document.

    Namespace-agnostic. Returns ``[]`` for malformed XML instead of raising.
    """
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8", errors="replace")
    if not xml_bytes or not xml_bytes.strip():
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    urls = []
    for element in root.iter():
        tag = element.tag
        if isinstance(tag, str) and tag.split("}")[-1] == "loc":
            text = (element.text or "").strip()
            if text:
                urls.append(text)
    return list(dict.fromkeys(urls))


def discover_feeds(html: str, base_url: str) -> list[str]:
    """Find RSS/Atom feed URLs from ``<link rel="alternate">`` tags in HTML."""
    if not html:
        return []
    feeds = []
    for tag_match in _LINK_TAG.finditer(html):
        tag = tag_match.group(0)
        if not _REL_ALTERNATE.search(tag):
            continue
        if not _FEED_TYPE.search(tag):
            continue
        href = _HREF.search(tag)
        if href:
            feeds.append(urljoin(base_url, href.group(1).strip()))
    return list(dict.fromkeys(feeds))
