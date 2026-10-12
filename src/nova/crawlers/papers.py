"""Research-paper-focused crawler: prefers arXiv / DOI links."""

from __future__ import annotations

from urllib.parse import urlparse

from nova.crawlers.base import BaseCrawler
from nova.crawlers.extract import extract_paper_metadata

__all__ = ["PapersCrawler"]


class PapersCrawler(BaseCrawler):
    """Crawler that stays on paper landing pages and PDFs.

    A discovered link is followed when it points at an arXiv abstract/PDF
    page, a ``doi.org`` link, or a ``.pdf`` URL. The paper title and abstract
    (heuristically extracted) are stored in the result's ``flags`` under
    ``"paper_title"`` and ``"paper_abstract"``.
    """

    def should_follow(self, url: str) -> bool:
        if not super().should_follow(url):
            return False
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        if host in ("arxiv.org", "export.arxiv.org"):
            return "/abs/" in path or "/pdf/" in path
        if host == "doi.org":
            return True
        return path.endswith(".pdf")

    def process_response(self, url: str, fetch_result, depth: int):
        result = super().process_response(url, fetch_result, depth)
        if result is None or result.error:
            return result
        content_type = (getattr(fetch_result, "content_type", "") or "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return result  # binary PDF: no HTML metadata to mine
        body = fetch_result.body or b""
        try:
            html = body.decode("utf-8", errors="replace")
            meta = extract_paper_metadata(html)
        except Exception:
            return result
        if meta.get("title"):
            result.flags["paper_title"] = meta["title"]
        if meta.get("abstract"):
            result.flags["paper_abstract"] = meta["abstract"]
        return result
