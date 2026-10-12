"""Sitemap and feed discovery helpers (stdlib only)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

__all__ = [
    "discover_sitemaps",
    "parse_sitemap",
    "parse_sitemap_detailed",
    "is_sitemap_index",
    "iter_sitemap_urls",
    "discover_feeds",
    "parse_feed",
]

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
    return [entry["url"] for entry in parse_sitemap_detailed(xml_bytes)]


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def parse_sitemap_detailed(xml_bytes: bytes | str) -> list[dict]:
    """Parse a sitemap / sitemap index into ``[{url, lastmod}]`` dicts.

    ``lastmod`` is the raw ``<lastmod>`` string or None. Namespace-agnostic;
    returns ``[]`` for malformed XML instead of raising. De-duplicated by URL,
    keeping the first (highest in document order) ``lastmod`` seen.
    """
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8", errors="replace")
    if not xml_bytes or not xml_bytes.strip():
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    entries: list[dict] = []
    seen: set[str] = set()
    # Both <url> and <sitemap> entries carry <loc> (+ optional <lastmod>).
    for element in root.iter():
        if _local_name(element.tag) not in ("url", "sitemap"):
            continue
        loc = lastmod = None
        for child in element:
            name = _local_name(child.tag)
            text = (child.text or "").strip()
            if name == "loc" and text:
                loc = text
            elif name == "lastmod" and text:
                lastmod = text
        if loc and loc not in seen:
            seen.add(loc)
            entries.append({"url": loc, "lastmod": lastmod})
    return entries


def is_sitemap_index(xml_bytes: bytes | str) -> bool:
    """True when the document is a ``<sitemapindex>`` (nested sitemaps)."""
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8", errors="replace")
    return _is_sitemap_index(xml_bytes)


def _is_sitemap_index(xml_bytes: bytes) -> bool:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return False
    return _local_name(root.tag) == "sitemapindex"


def iter_sitemap_urls(fetcher, sitemap_url: str, *, max_levels: int = 3,
                      _seen: set[str] | None = None,
                      _level: int = 0) -> list[dict]:
    """Recursively resolve a sitemap (index) into ``[{url, lastmod}]`` page entries.

    Nested ``<sitemapindex>`` documents are followed through ``fetcher``
    (sandbox safety checks apply to every request). ``max_levels`` bounds the
    recursion. Never raises: fetch/parse failures yield ``[]`` for that branch.
    """
    if _seen is None:
        _seen = set()
    if sitemap_url in _seen or _level > max_levels:
        return []
    _seen.add(sitemap_url)
    try:
        result = fetcher.fetch(sitemap_url)
        body = result.body or b""
        if not body:
            return []
    except Exception:
        return []
    if _is_sitemap_index(body):
        nested: list[dict] = []
        for entry in parse_sitemap_detailed(body):
            nested.extend(iter_sitemap_urls(
                fetcher, entry["url"], max_levels=max_levels,
                _seen=_seen, _level=_level + 1))
        return nested
    return parse_sitemap_detailed(body)


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


def _parse_pub_date(raw: str | None) -> str | None:
    """Normalize a feed date to ISO-8601; None when unparsable."""
    if not raw:
        return None
    text = raw.strip()
    # ISO-8601 first (Atom).
    try:
        iso = text.replace("Z", "+00:00")
        from datetime import datetime

        return datetime.fromisoformat(iso).isoformat()
    except Exception:
        pass
    # RFC-2822 (RSS pubDate).
    try:
        return parsedate_to_datetime(text).isoformat()
    except Exception:
        return None


def parse_feed(xml_bytes: bytes | str) -> list[dict]:
    """Parse an RSS/Atom feed into ``[{url, published}]`` dicts.

    ``published`` is an ISO-8601 string or None. Results are ordered freshest
    first (undated entries last). Returns ``[]`` for malformed XML.
    """
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8", errors="replace")
    if not xml_bytes or not xml_bytes.strip():
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    items: list[dict] = []
    seen: set[str] = set()

    def _text(element, names: set[str]) -> str | None:
        for child in element:
            if _local_name(child.tag) in names and (child.text or "").strip():
                return child.text.strip()
        return None

    for element in root.iter():
        name = _local_name(element.tag)
        if name == "item":  # RSS
            url = _text(element, {"link", "guid"})
            published = _parse_pub_date(_text(element, {"pubDate", "date"}))
        elif name == "entry":  # Atom
            url = None
            published = None
            for child in element:
                child_name = _local_name(child.tag)
                if child_name == "link" and url is None:
                    rel = (child.attrib.get("rel") or "").lower()
                    href = (child.attrib.get("href") or "").strip()
                    if href and rel in ("", "alternate"):
                        url = href
                elif child_name in ("published", "updated") and published is None:
                    published = _parse_pub_date(child.text)
            if url is None:  # fall back to <id> when no link is present
                url = _text(element, {"id"})
        else:
            continue
        if url and url not in seen:
            seen.add(url)
            items.append({"url": url, "published": published})

    items.sort(key=lambda item: (item["published"] is None,
                                 item["published"] or ""))
    # Freshest first: reverse the dated prefix while keeping undated last.
    dated = [i for i in items if i["published"]][::-1]
    undated = [i for i in items if not i["published"]]
    return dated + undated
