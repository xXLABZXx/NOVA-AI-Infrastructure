"""Base crawler: concurrent, polite crawling with a persistent frontier.

:class:`BaseCrawler` fetches through the sandbox HTTP client inside a
:class:`~nova.crawlers.scheduler.CrawlScheduler` thread pool, so every
request keeps the sandbox safety checks (SSRF guard, allow/blocklists, size
caps). Politeness is per-host (robots.txt ``crawl-delay`` or the configured
default), retries use exponential backoff with jitter, and a SQLite frontier
makes crawls resumable.

The public surface is unchanged: ``crawl(seeds) -> list[CrawlResult]``,
``from_config``, ``should_follow`` / ``extract_content`` /
``process_response`` hooks, ``save_results`` and the ``failed`` list.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from nova.crawlers.canonical import canonicalize_url
from nova.crawlers.discovery import (
    discover_feeds,
    is_sitemap_index,
    parse_feed,
    parse_sitemap_detailed,
)
from nova.crawlers.extract import extract_article
from nova.crawlers.frontier import Frontier, compute_priority
from nova.crawlers.robots import RobotsChecker
from nova.crawlers.scheduler import CrawlScheduler, backoff_delay

log = logging.getLogger(__name__)

__all__ = ["CrawlResult", "BaseCrawler"]


@dataclass
class CrawlResult:
    """Outcome of fetching and processing a single URL."""

    url: str
    status_code: int
    title: str
    text: str
    lang: str
    links: list[str] = field(default_factory=list)
    content_hash: str = ""
    depth: int = 0
    error: str | None = None
    retrieved_at: str = ""
    flags: dict = field(default_factory=dict)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class _HttpStatusError(Exception):
    """Synthetic exception for retryable HTTP statuses (429/5xx)."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def _is_transient(exc: BaseException) -> bool:
    """Best-effort classification of an exception as a retryable network error."""
    name = type(exc).__name__.lower()
    transient_markers = (
        "timeout", "timedout", "connection", "connect", "network",
        "temporary", "unreachable", "reset", "refused", "dns",
        "ssl", "socket", "http", "retry", "ratelimit", "rate_limit",
        "toomanyrequests", "serviceunavailable", "badgateway",
        "gatewaytimeout", "oserror", "ioerror",
    )
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return any(marker in name for marker in transient_markers)


def _parse_http_date(raw: str | None) -> float | None:
    """Parse an HTTP / ISO date to epoch seconds; None when unparsable."""
    if not raw:
        return None
    text = raw.strip()
    try:
        iso = text.replace("Z", "+00:00")
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(text)
        if dt is not None:
            return dt.timestamp()
    except Exception:
        pass
    return None


class BaseCrawler:
    """Concurrent polite crawler built on a persistent URL frontier.

    Parameters
    ----------
    fetcher:
        Object with ``.fetch(url)`` returning a result with ``status_code``,
        ``headers``, ``body`` and ``content_type`` (may raise on failure).
        When ``fetch`` also accepts a ``headers`` keyword, conditional
        ``If-None-Match`` / ``If-Modified-Since`` headers from the seen store
        are sent for revalidation.
    seen_store:
        :class:`SeenStore` used to skip already-crawled URLs (may be None).
    frontier:
        :class:`Frontier` work queue. When None, an ephemeral in-memory
        frontier is used per :meth:`crawl` call; pass a file-backed one (as
        ``runner.py`` does) to make crawls resumable across restarts.
    user_agent, max_depth, max_pages_per_domain, politeness_delay_seconds,
    respect_robots_txt, retry_limit:
        Crawl behaviour knobs (as before). ``max_pages_per_domain`` is the
        legacy name for ``max_pages_per_host``.
    concurrency:
        Bounded worker-thread count for fetching.
    per_host_delay_seconds:
        Default minimum seconds between requests to the same host
        (robots.txt ``crawl-delay`` wins when declared). Falls back to
        ``politeness_delay_seconds``.
    robots_ttl_hours:
        How long a cached robots.txt stays valid.
    max_pages_total:
        Global cap on fetched pages per :meth:`crawl` (``0`` = unlimited).
    retry_backoff_base_seconds / retry_backoff_max_seconds:
        Exponential backoff with jitter between retries.
    requests_per_minute:
        Global scheduler token-bucket rate (``0`` disables; the sandbox
        client enforces its own limit regardless).
    follow_sitemaps / follow_feeds:
        Enqueue ``Sitemap:`` URLs from robots.txt and RSS/Atom feeds found
        in pages.
    """

    def __init__(
        self,
        *,
        fetcher,
        seen_store=None,
        frontier=None,
        user_agent: str = "NOVA-ResearchBot/0.1",
        max_depth: int = 2,
        max_pages_per_domain: int = 200,
        max_pages_per_host: int | None = None,
        max_pages_total: int = 5000,
        politeness_delay_seconds: float = 1.0,
        per_host_delay_seconds: float | None = None,
        crawl_delay_default_seconds: float | None = None,
        robots_ttl_hours: float = 24.0,
        respect_robots_txt: bool = True,
        retry_limit: int = 2,
        retry_backoff_base_seconds: float = 0.5,
        retry_backoff_max_seconds: float = 60.0,
        requests_per_minute: float = 120,
        concurrency: int = 10,
        follow_sitemaps: bool = True,
        follow_feeds: bool = True,
    ):
        self.fetcher = fetcher
        self.seen_store = seen_store
        self._frontier = frontier
        self.user_agent = user_agent
        self.max_depth = max_depth
        self.max_pages_per_domain = max_pages_per_domain
        self.max_pages_per_host = (
            max_pages_per_domain if max_pages_per_host is None
            else max_pages_per_host
        )
        self.max_pages_total = max_pages_total
        self.politeness_delay_seconds = politeness_delay_seconds
        self.per_host_delay_seconds = (
            politeness_delay_seconds if per_host_delay_seconds is None
            else per_host_delay_seconds
        )
        self.crawl_delay_default_seconds = (
            self.per_host_delay_seconds if crawl_delay_default_seconds is None
            else crawl_delay_default_seconds
        )
        self.robots_ttl_hours = robots_ttl_hours
        self.respect_robots_txt = respect_robots_txt
        self.retry_limit = retry_limit
        self.retry_backoff_base_seconds = retry_backoff_base_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self.requests_per_minute = requests_per_minute
        self.concurrency = concurrency
        self.follow_sitemaps = follow_sitemaps
        self.follow_feeds = follow_feeds
        self.robots = (
            RobotsChecker(user_agent=user_agent, fetcher=fetcher,
                          ttl_hours=robots_ttl_hours,
                          cache=frontier)
            if respect_robots_txt
            else None
        )
        #: Failures recorded as {"url", "error", "attempts"} dicts.
        self.failed: list[dict] = []
        #: Dead-letter entries for this run: {"url","reason","attempts","error","at"}.
        self.dead: list[dict] = []
        #: Run counters (retried, dead, not_modified, ...).
        self.stats: dict[str, int] = {}
        self._results: list[CrawlResult] = []
        self._results_lock = threading.Lock()
        self._sitemap_primed: set[str] = set()

    @classmethod
    def from_config(cls, section: dict, *, fetcher, seen_store=None,
                    frontier=None) -> "BaseCrawler":
        """Build a crawler from a config section dict (see configs/config.yaml)."""
        per_host_delay = float(section.get(
            "per_host_delay_seconds",
            section.get("politeness_delay_seconds", 1.0)))
        return cls(
            fetcher=fetcher,
            seen_store=seen_store,
            frontier=frontier,
            user_agent=section.get("user_agent", "NOVA-ResearchBot/0.1"),
            max_depth=int(section.get("max_depth", 2)),
            max_pages_per_domain=int(section.get("max_pages_per_domain", 200)),
            max_pages_per_host=int(section.get(
                "max_pages_per_host",
                section.get("max_pages_per_domain", 200))),
            max_pages_total=int(section.get("max_pages_total", 5000)),
            politeness_delay_seconds=float(
                section.get("politeness_delay_seconds", 1.0)),
            per_host_delay_seconds=per_host_delay,
            crawl_delay_default_seconds=float(
                section.get("crawl_delay_default_seconds", per_host_delay)),
            robots_ttl_hours=float(section.get("robots_ttl_hours", 24.0)),
            respect_robots_txt=bool(section.get("respect_robots_txt", True)),
            retry_limit=int(section.get("retry_limit", 2)),
            retry_backoff_base_seconds=float(
                section.get("retry_backoff_base_seconds", 0.5)),
            retry_backoff_max_seconds=float(
                section.get("retry_backoff_max_seconds", 60.0)),
            requests_per_minute=float(section.get("requests_per_minute", 120)),
            concurrency=int(section.get("concurrency", 10)),
            follow_sitemaps=bool(section.get("follow_sitemaps", True)),
            follow_feeds=bool(section.get("follow_feeds", True)),
        )

    # -- hooks for subclasses ---------------------------------------------
    def should_follow(self, url: str) -> bool:
        """Whether a *discovered* link may be crawled. Seeds always run."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    def extract_content(self, url: str, fetch_result) -> dict:
        """Turn a fetch result into article content.

        Returns ``{title, text, description, publish_date, author, lang,
        lang_confidence, links, url, quality}`` via readability-style
        main-content extraction.
        """
        body = fetch_result.body or b""
        charset = "utf-8"
        content_type = (getattr(fetch_result, "content_type", "") or "")
        if "charset=" in content_type.lower():
            charset = content_type.lower().split("charset=")[-1].split(";")[0].strip()
        try:
            html = body.decode(charset, errors="replace")
        except (LookupError, ValueError):
            html = body.decode("utf-8", errors="replace")
        return extract_article(html, url)

    # -- main crawl entry point -----------------------------------------------
    def crawl(self, seeds: list[str]) -> list[CrawlResult]:
        """Crawl starting from ``seeds`` using the concurrent scheduler.

        Never raises because of a single URL: per-URL failures are captured in
        the returned :class:`CrawlResult` (``error`` set), in ``self.failed``
        and — for permanently failed URLs — in ``self.dead`` / the frontier's
        dead-letter records.
        """
        self.failed = []
        self.dead = []
        self.stats = {"dispatched": 0, "completed": 0, "skipped": 0,
                      "retried": 0, "dead": 0, "not_modified": 0}
        self._results = []
        self._sitemap_primed = set()

        own_frontier = self._frontier is None
        frontier = self._frontier if not own_frontier else Frontier(":memory:")
        self._crawl_frontier = frontier
        try:
            for index, seed in enumerate(seeds):
                if not seed:
                    continue
                canon = canonicalize_url(seed)
                if not canon:
                    continue
                frontier.add(
                    canon, depth=0, discovered_from=None,
                    priority=compute_priority(0, seed_index=index),
                    max_attempts=1 + self.retry_limit,
                )
            scheduler = CrawlScheduler(
                max_workers=self.concurrency,
                default_host_delay=self.per_host_delay_seconds,
                requests_per_minute=self.requests_per_minute,
                max_pages_total=self.max_pages_total,
            )
            run_stats = scheduler.run(frontier, self)
            for key in ("dispatched", "completed", "skipped"):
                self.stats[key] = run_stats.get(key, 0)
        finally:
            self._crawl_frontier = None
            if own_frontier:
                frontier.close()
        return self._results

    # -- scheduler handler protocol -------------------------------------------
    def scheduler_prepare(self, row: dict) -> tuple[bool, str]:
        """Decide whether a claimed frontier row may be dispatched."""
        url = row["url"]
        depth = row["depth"]
        if depth > self.max_depth:
            return False, "depth"
        is_seed = depth == 0 and not row.get("discovered_from")
        if not is_seed:
            try:
                if not self.should_follow(url):
                    return False, "filtered"
            except Exception:
                return False, "filtered"
        if self.seen_store is not None:
            try:
                if self.seen_store.has(url) and not row.get("revalidate"):
                    return False, "seen"
            except Exception:
                pass
        if self.robots is not None:
            try:
                if not self.robots.can_fetch(url):
                    return False, "robots"
            except Exception:
                pass
            # Prime sitemap discovery once per host.
            if self.follow_sitemaps:
                host = row["host"]
                if host not in self._sitemap_primed:
                    self._sitemap_primed.add(host)
                    try:
                        for sitemap_url in self.robots.sitemaps(url):
                            self._enqueue_link(
                                sitemap_url, 0, f"robots.txt:{host}",
                                kind="sitemap", boost=1.0)
                    except Exception as exc:
                        log.debug("sitemap priming failed for %s: %s", host, exc)
        return True, ""

    def scheduler_host_delay(self, host: str) -> float:
        """Per-host delay: robots.txt crawl-delay wins, else the default."""
        if self.robots is not None:
            try:
                delay = self.robots.crawl_delay(f"http://{host}/")
                if delay is not None:
                    return max(0.0, float(delay))
            except Exception:
                pass
        return self.crawl_delay_default_seconds

    def scheduler_process(self, row: dict) -> None:
        """Fetch one URL and handle success / retry / dead-letter (worker thread)."""
        url = row["url"]
        depth = row["depth"]
        kind = row.get("kind") or "page"
        try:
            fetch_result = self._fetch_with_conditional(url)
        except Exception as exc:
            self._handle_fetch_error(row, exc)
            return

        status = getattr(fetch_result, "status_code", 0) or 0
        if status == 304:
            self._handle_not_modified(row)
            return
        if status == 429 or 500 <= status <= 599:
            self._handle_fetch_error(
                row, _HttpStatusError(status), http_status=status)
            return
        if not 200 <= status < 300:
            result = CrawlResult(
                url=url, status_code=status, title="", text="",
                lang="unknown", depth=depth, error=f"HTTP {status}",
                retrieved_at=_utcnow(), flags={"http_status": status},
            )
            self._finish_page(row, result, fetch_result, ok=False)
            return

        discovery_kind = (kind if kind in ("sitemap", "feed")
                          else self._discovery_kind(url, fetch_result))
        if discovery_kind in ("sitemap", "feed"):
            self._handle_discovery(row, fetch_result, discovery_kind)
            return

        try:
            result = self.process_response(url, fetch_result, depth)
        except Exception as exc:  # one bad page must not kill the crawl
            message = f"{type(exc).__name__}: {exc}"
            log.debug("processing failed for %s: %s", url, message)
            result = CrawlResult(
                url=url, status_code=status, title="", text="",
                lang="unknown", depth=depth, error=message,
                retrieved_at=_utcnow(),
            )
            self._finish_page(row, result, fetch_result, ok=False)
            self._record_process_failure(row, message)
            return
        self._finish_page(row, result, fetch_result, ok=True)

    def scheduler_crash(self, row: dict, exc: BaseException) -> None:
        """Last resort: route an unexpected worker exception into retries."""
        log.debug("worker crash for %s: %s", row.get("url"), exc)
        self._handle_fetch_error(row, exc)

    # -- fetch helpers ----------------------------------------------------------
    def _fetch_with_conditional(self, url: str):
        """Fetch, sending conditional headers when the fetcher supports them."""
        headers: dict[str, str] = {}
        if self.seen_store is not None:
            try:
                headers = self.seen_store.conditional_headers(url)
            except Exception:
                headers = {}
        fetch = self.fetcher.fetch
        if headers:
            try:
                if "headers" in inspect.signature(fetch).parameters:
                    return fetch(url, headers=headers)
            except (ValueError, TypeError):
                pass
        return fetch(url)

    @staticmethod
    def _discovery_kind(url: str, fetch_result) -> str | None:
        content_type = (getattr(fetch_result, "content_type", "") or "").lower()
        path = urlparse(url).path.lower()
        if ("rss" in content_type or "atom" in content_type
                or path.endswith((".rss", ".atom"))):
            return "feed"
        if "xml" in content_type or path.endswith(".xml"):
            return "sitemap"
        return None

    @staticmethod
    def _decode_body(fetch_result) -> str:
        body = fetch_result.body or b""
        content_type = (getattr(fetch_result, "content_type", "") or "")
        charset = "utf-8"
        if "charset=" in content_type.lower():
            charset = content_type.lower().split("charset=")[-1].split(";")[0].strip()
        try:
            return body.decode(charset, errors="replace")
        except (LookupError, ValueError):
            return body.decode("utf-8", errors="replace")

    # -- outcome handling ---------------------------------------------------------
    def _finish_page(self, row: dict, result: CrawlResult, fetch_result,
                     *, ok: bool) -> None:
        url = row["url"]
        frontier = self._active_frontier()
        with self._results_lock:
            self._results.append(result)
        if frontier is not None:
            frontier.mark_done(url)
            frontier.inc_host_pages(row["host"])
            frontier.log_attempt(url, row["attempts"] + 1,
                                 "ok" if ok else "http_error",
                                 result.error or "")
        if ok and not result.error:
            if self.seen_store is not None:
                try:
                    headers = getattr(fetch_result, "headers", {}) or {}
                    self.seen_store.mark(
                        url,
                        etag=headers.get("ETag") or headers.get("etag"),
                        last_modified=headers.get("Last-Modified")
                        or headers.get("last-modified"),
                    )
                except Exception as exc:
                    log.debug("seen_store.mark failed for %s: %s", url, exc)
            # Follow page links (depth-gated) and discover feeds.
            if row["depth"] < self.max_depth:
                for link in result.links:
                    self._enqueue_link(link, row["depth"] + 1, url)
            if self.follow_feeds:
                try:
                    html = self._decode_body(fetch_result)
                    for feed_url in discover_feeds(html, url):
                        self._enqueue_link(feed_url, row["depth"], url,
                                           kind="feed", boost=0.5)
                except Exception as exc:
                    log.debug("feed discovery failed for %s: %s", url, exc)

    def _handle_not_modified(self, row: dict) -> None:
        """304: content unchanged — record without re-extracting."""
        frontier = self._active_frontier()
        if frontier is not None:
            frontier.mark_done(row["url"])
            frontier.log_attempt(row["url"], row["attempts"] + 1,
                                 "not_modified", "304 Not Modified")
        with self._results_lock:
            self.stats["not_modified"] = self.stats.get("not_modified", 0) + 1

    def _handle_fetch_error(self, row: dict, exc: BaseException,
                            *, http_status: int | None = None) -> None:
        url = row["url"]
        host = row["host"]
        attempts_done = row["attempts"] + 1
        if http_status is not None:
            message = f"HTTP {http_status}"
            transient = http_status == 429 or 500 <= http_status <= 599
        else:
            message = f"{type(exc).__name__}: {exc}"
            transient = _is_transient(exc)
        log.debug("fetch failed for %s (attempt %d): %s", url, attempts_done,
                  message)
        frontier = self._active_frontier()
        if frontier is not None:
            frontier.log_attempt(
                url, attempts_done,
                "transient_error" if transient else "fatal_error", message)
        # Total attempts allowed: 1 initial + retry_limit retries.
        if transient and attempts_done <= self.retry_limit:
            delay = backoff_delay(attempts_done,
                                  self.retry_backoff_base_seconds,
                                  self.retry_backoff_max_seconds)
            if frontier is not None:
                frontier.mark_retry(url,
                                    next_fetch_at=time.monotonic() + delay,
                                    error=message, attempts=attempts_done)
                if http_status is not None:
                    # 429/5xx backs off the whole host, not just this URL.
                    frontier.set_host_backoff(host, time.monotonic() + delay)
            with self._results_lock:
                self.stats["retried"] = self.stats.get("retried", 0) + 1
            return
        # Permanently failed: dead-letter.
        reason = "max_attempts_exceeded" if transient else "fatal_error"
        if frontier is not None:
            frontier.mark_dead(url, reason=reason, error=message,
                               attempts=attempts_done)
        entry = {"url": url, "reason": reason, "attempts": attempts_done,
                 "error": message, "at": _utcnow()}
        result = CrawlResult(
            url=url, status_code=http_status or 0, title="", text="",
            lang="unknown", depth=row["depth"], error=message,
            retrieved_at=_utcnow(), flags={"dead": True, "reason": reason},
        )
        with self._results_lock:
            self.dead.append(entry)
            self.failed.append({"url": url, "error": message,
                                "attempts": attempts_done})
            self._results.append(result)
            self.stats["dead"] = self.stats.get("dead", 0) + 1

    def _record_process_failure(self, row: dict, message: str) -> None:
        frontier = self._active_frontier()
        if frontier is not None:
            frontier.log_attempt(row["url"], row["attempts"] + 1,
                                 "process_error", message)
        with self._results_lock:
            self.failed.append({"url": row["url"], "error": message,
                                "attempts": row["attempts"] + 1})

    def _handle_discovery(self, row: dict, fetch_result, kind: str) -> None:
        """Parse a sitemap / feed and enqueue what it points at."""
        url = row["url"]
        body = fetch_result.body or b""
        discovered = 0
        try:
            if kind == "feed":
                for item in parse_feed(body):
                    boost = 1.0 if item["published"] else 0.3
                    if self._enqueue_link(item["url"], row["depth"], url,
                                          boost=boost):
                        discovered += 1
            elif is_sitemap_index(body) and row["slevel"] < 3:
                for entry in parse_sitemap_detailed(body):
                    if self._enqueue_link(entry["url"], row["depth"], url,
                                          kind="sitemap",
                                          slevel=row["slevel"] + 1, boost=0.8):
                        discovered += 1
            else:
                for entry in parse_sitemap_detailed(body):
                    if self._enqueue_link(entry["url"], row["depth"], url,
                                          lastmod=entry["lastmod"], boost=0.5):
                        discovered += 1
        except Exception as exc:
            log.debug("discovery parse failed for %s: %s", url, exc)
        result = CrawlResult(
            url=url,
            status_code=getattr(fetch_result, "status_code", 0) or 0,
            title="", text="", lang="unknown", depth=row["depth"],
            retrieved_at=_utcnow(),
            flags={"discovery": kind, "discovered_urls": discovered},
        )
        self._finish_page(row, result, fetch_result, ok=True)

    # -- enqueue ------------------------------------------------------------------
    def _active_frontier(self):
        return getattr(self, "_crawl_frontier", None)

    def _enqueue_link(self, url: str, depth: int, discovered_from: str | None,
                      *, kind: str = "page", boost: float = 0.0,
                      lastmod: str | None = None, slevel: int = 0) -> bool:
        canon = canonicalize_url(url)
        if not canon:
            return False
        parsed = urlparse(canon)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        is_seed = depth == 0 and not discovered_from
        if not is_seed:
            try:
                if not self.should_follow(canon):
                    return False
            except Exception:
                return False
        # Incremental: skip sitemap URLs whose <lastmod> is not newer than
        # what we already crawled; revalidate ones that changed.
        revalidate = False
        if lastmod and self.seen_store is not None:
            try:
                if self.seen_store.has(canon):
                    if self._lastmod_not_newer(lastmod, canon):
                        return False
                    revalidate = True
            except Exception:
                pass
        frontier = self._active_frontier()
        if frontier is None:
            return False
        return frontier.add(
            canon, depth=depth, discovered_from=discovered_from,
            priority=compute_priority(depth, boost=boost), kind=kind,
            slevel=slevel, max_attempts=1 + self.retry_limit, lastmod=lastmod,
            revalidate=revalidate,
        )

    def _lastmod_not_newer(self, lastmod: str, url: str) -> bool:
        """True when the sitemap ``lastmod`` is not newer than our crawl."""
        seen_at = self.seen_store.seen_at(url)
        if not seen_at:
            return False
        lastmod_ts = _parse_http_date(lastmod)
        seen_ts = _parse_http_date(seen_at)
        if lastmod_ts is None or seen_ts is None:
            return False
        return lastmod_ts <= seen_ts

    # -- response processing --------------------------------------------------------
    def process_response(self, url: str, fetch_result, depth: int) -> CrawlResult | None:
        """Convert a successful fetch into a :class:`CrawlResult`.

        Returns ``None`` for "not modified" style responses that carry no
        content to process.
        """
        status = getattr(fetch_result, "status_code", 0) or 0
        if status == 304:
            return None
        if not 200 <= status < 300:
            return CrawlResult(
                url=url, status_code=status, title="", text="", lang="unknown",
                depth=depth, error=f"HTTP {status}", retrieved_at=_utcnow(),
            )

        extracted = self.extract_content(url, fetch_result)
        text = extracted.get("text", "")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        links: list[str] = []
        for raw_link in extracted.get("links", []):
            try:
                absolute = urljoin(url, raw_link)
            except Exception:
                continue
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            links.append(absolute)
        # De-duplicate while preserving order.
        links = list(dict.fromkeys(links))

        flags: dict = {
            "quality": extracted.get("quality", {}),
        }
        for key in ("description", "publish_date", "author"):
            if extracted.get(key):
                flags[key] = extracted[key]
        guard_text = self._load_injection_guard()
        if guard_text is not None and text:
            try:
                outcome = guard_text(text)
                # The guard may return a plain bool or a result object
                # with a ``flagged`` attribute (nova.security GuardResult).
                flagged = (outcome.flagged if hasattr(outcome, "flagged")
                           else bool(outcome))
                flags["injection_flagged"] = bool(flagged)
            except Exception as exc:
                log.debug("injection guard failed for %s: %s", url, exc)

        return CrawlResult(
            url=url,
            status_code=status,
            title=extracted.get("title", ""),
            text=text,
            lang=extracted.get("lang", "unknown"),
            links=links,
            content_hash=content_hash,
            depth=depth,
            error=None,
            retrieved_at=_utcnow(),
            flags=flags,
        )

    @staticmethod
    def _load_injection_guard():
        """Lazily import the prompt-injection guard; None when unavailable."""
        try:
            from nova.security.injection_guard import guard_text
            return guard_text
        except Exception:
            return None

    # -- persistence ----------------------------------------------------------
    def save_results(self, results: list[CrawlResult], out_path: str | Path,
                     *, mode: str = "w") -> Path:
        """Write results as JSONL: one doc per line.

        Each line: ``{id, url, title, text, lang, retrieved_at, sha256,
        source, quality, description, publish_date, author}`` where ``source``
        is the crawler class name. ``mode="a"`` appends (used when resuming a
        crawl); the default ``"w"`` overwrites.
        """
        out_path = Path(out_path)
        if str(out_path.parent):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        source = type(self).__name__
        with open(out_path, mode, encoding="utf-8") as fh:
            for result in results:
                flags = result.flags or {}
                doc = {
                    "id": uuid.uuid4().hex,
                    "url": result.url,
                    "title": result.title,
                    "text": result.text,
                    "lang": result.lang,
                    "retrieved_at": result.retrieved_at or _utcnow(),
                    "sha256": result.content_hash,
                    "source": source,
                    "quality": flags.get("quality", {}),
                    "description": flags.get("description", ""),
                    "publish_date": flags.get("publish_date", ""),
                    "author": flags.get("author", ""),
                }
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return out_path

    def dead_letter_entries(self) -> list[dict]:
        """Dead-letter records for permanently failed URLs this run."""
        frontier = self._active_frontier()
        if frontier is not None:
            try:
                entries = frontier.dead_entries()
                if entries:
                    return entries
            except Exception:
                pass
        return list(self.dead)
