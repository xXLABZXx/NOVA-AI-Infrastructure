"""Base crawler: polite BFS crawl with retries, robots support and JSONL output."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from nova.crawlers.extract import extract_text
from nova.crawlers.robots import RobotsChecker

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


class BaseCrawler:
    """Polite breadth-first crawler.

    Parameters
    ----------
    fetcher:
        Object with ``.fetch(url)`` returning a result with ``status_code``,
        ``headers``, ``body`` and ``content_type`` (may raise on failure).
    seen_store:
        :class:`SeenStore` used to skip already-crawled URLs (may be None).
    user_agent, max_depth, max_pages_per_domain, politeness_delay_seconds,
    respect_robots_txt, retry_limit:
        Crawl behaviour knobs. ``retry_limit`` is the number of *retries*
        after the first attempt (total attempts = 1 + retry_limit).
    """

    def __init__(
        self,
        *,
        fetcher,
        seen_store=None,
        user_agent: str = "NOVA-ResearchBot/0.1",
        max_depth: int = 2,
        max_pages_per_domain: int = 200,
        politeness_delay_seconds: float = 1.0,
        respect_robots_txt: bool = True,
        retry_limit: int = 2,
    ):
        self.fetcher = fetcher
        self.seen_store = seen_store
        self.user_agent = user_agent
        self.max_depth = max_depth
        self.max_pages_per_domain = max_pages_per_domain
        self.politeness_delay_seconds = politeness_delay_seconds
        self.respect_robots_txt = respect_robots_txt
        self.retry_limit = retry_limit
        self.robots = (
            RobotsChecker(user_agent=user_agent, fetcher=fetcher)
            if respect_robots_txt
            else None
        )
        #: Failures recorded as {"url", "error", "attempts"} dicts.
        self.failed: list[dict] = []
        self._last_fetch_at: dict[str, float] = {}

    @classmethod
    def from_config(cls, section: dict, *, fetcher, seen_store=None) -> "BaseCrawler":
        """Build a crawler from a config section dict (see configs/config.yaml)."""
        return cls(
            fetcher=fetcher,
            seen_store=seen_store,
            user_agent=section.get("user_agent", "NOVA-ResearchBot/0.1"),
            max_depth=int(section.get("max_depth", 2)),
            max_pages_per_domain=int(section.get("max_pages_per_domain", 200)),
            politeness_delay_seconds=float(
                section.get("politeness_delay_seconds", 1.0)),
            respect_robots_txt=bool(section.get("respect_robots_txt", True)),
            retry_limit=int(section.get("retry_limit", 2)),
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
        """Turn a fetch result into ``{title, text, lang, links}``."""
        body = fetch_result.body or b""
        charset = "utf-8"
        content_type = (getattr(fetch_result, "content_type", "") or "")
        if "charset=" in content_type.lower():
            charset = content_type.lower().split("charset=")[-1].split(";")[0].strip()
        try:
            html = body.decode(charset, errors="replace")
        except (LookupError, ValueError):
            html = body.decode("utf-8", errors="replace")
        return extract_text(html, url)

    # -- main crawl loop ----------------------------------------------------
    def crawl(self, seeds: list[str]) -> list[CrawlResult]:
        """Breadth-first crawl starting from ``seeds``.

        Never raises because of a single URL: per-URL failures are captured in
        the returned :class:`CrawlResult` (``error`` set) and in ``self.failed``.
        """
        self.failed = []
        self._last_fetch_at = {}
        results: list[CrawlResult] = []
        domain_counts: dict[str, int] = {}
        frontier: deque[tuple[str, int]] = deque(
            (seed, 0) for seed in seeds if seed
        )

        while frontier:
            url, depth = frontier.popleft()

            if depth > self.max_depth:
                continue
            # Seeds are explicit user intent; discovered links must pass the hook.
            if depth > 0 and not self.should_follow(url):
                continue
            if self.seen_store is not None and self.seen_store.has(url):
                continue
            if self.robots is not None and not self.robots.can_fetch(url):
                log.debug("robots.txt disallows %s; skipping", url)
                continue
            domain = urlparse(url).netloc.lower()
            if domain_counts.get(domain, 0) >= self.max_pages_per_domain:
                continue

            self._polite_wait(domain)

            fetch_result, attempts, fetch_error = self._fetch_with_retries(url)
            if fetch_error is not None:
                result = CrawlResult(
                    url=url, status_code=0, title="", text="", lang="unknown",
                    depth=depth, error=fetch_error,
                    retrieved_at=_utcnow(),
                )
                results.append(result)
                self.failed.append(
                    {"url": url, "error": fetch_error, "attempts": attempts}
                )
                continue

            try:
                result = self.process_response(url, fetch_result, depth)
            except Exception as exc:  # one bad page must not kill the crawl
                message = f"{type(exc).__name__}: {exc}"
                log.debug("processing failed for %s: %s", url, message)
                result = CrawlResult(
                    url=url,
                    status_code=getattr(fetch_result, "status_code", 0) or 0,
                    title="", text="", lang="unknown",
                    depth=depth, error=message, retrieved_at=_utcnow(),
                )
                results.append(result)
                self.failed.append({"url": url, "error": message, "attempts": 1})
                continue

            if result is None:  # e.g. 304 Not Modified
                continue

            results.append(result)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if self.seen_store is not None and not result.error:
                headers = getattr(fetch_result, "headers", {}) or {}
                self.seen_store.mark(
                    url,
                    etag=headers.get("ETag") or headers.get("etag"),
                    last_modified=headers.get("Last-Modified")
                    or headers.get("last-modified"),
                )
            if depth < self.max_depth:
                for link in result.links:
                    frontier.append((link, depth + 1))

        return results

    # -- internals ------------------------------------------------------------
    def _polite_wait(self, domain: str) -> None:
        delay = self.politeness_delay_seconds
        if delay <= 0:
            self._last_fetch_at[domain] = time.monotonic()
            return
        now = time.monotonic()
        last = self._last_fetch_at.get(domain, 0.0)
        wait = delay - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch_at[domain] = time.monotonic()

    def _fetch_with_retries(self, url: str):
        """Fetch with retries. Returns (result, attempts, error_message|None)."""
        attempts = 0
        last_error: BaseException | None = None
        while attempts <= self.retry_limit:
            attempts += 1
            try:
                return self.fetcher.fetch(url), attempts, None
            except Exception as exc:
                last_error = exc
                log.debug("fetch attempt %d for %s failed: %s",
                          attempts, url, exc)
                if not _is_transient(exc) or attempts > self.retry_limit:
                    break
                time.sleep(min(0.5 * attempts, 5.0))
        assert last_error is not None
        return None, attempts, f"{type(last_error).__name__}: {last_error}"

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

        flags: dict = {}
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
    def save_results(self, results: list[CrawlResult], out_path: str | Path) -> Path:
        """Write results as JSONL: one doc per line.

        Each line: ``{id, url, title, text, lang, retrieved_at, sha256, source}``
        where ``source`` is the crawler class name.
        """
        out_path = Path(out_path)
        if str(out_path.parent):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        source = type(self).__name__
        with open(out_path, "w", encoding="utf-8") as fh:
            for result in results:
                doc = {
                    "id": uuid.uuid4().hex,
                    "url": result.url,
                    "title": result.title,
                    "text": result.text,
                    "lang": result.lang,
                    "retrieved_at": result.retrieved_at or _utcnow(),
                    "sha256": result.content_hash,
                    "source": source,
                }
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return out_path
