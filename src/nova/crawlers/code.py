"""Source-code-focused crawler: prefers raw source files and code hosts."""

from __future__ import annotations

from urllib.parse import urlparse

from nova.crawlers.base import BaseCrawler
from nova.crawlers.extract import extract_code_text

__all__ = ["CodeCrawler"]

#: File extensions treated as source code (compared against the URL path).
_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
        ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb",
        ".php", ".swift", ".scala", ".sh", ".md", ".rst", ".toml",
        ".yaml", ".yml", ".json",
    }
)

#: Hosts known to serve raw source files or code listings.
_CODE_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
        "gist.githubusercontent.com",
        "gitlab.com",  # raw file URLs live under /-/raw/
    }
)


class CodeCrawler(BaseCrawler):
    """Crawler that prefers source files.

    A discovered link is followed when its path ends with a source-code
    extension or its host is a known code host (e.g.
    ``raw.githubusercontent.com``). Extracted text keeps ``<pre>``/``<code>``
    blocks verbatim so indentation survives.
    """

    def __init__(self, *, code_hosts: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.code_hosts = {h.lower() for h in
                           list(_CODE_HOSTS) + list(code_hosts or [])}

    @classmethod
    def from_config(cls, section: dict, *, fetcher, seen_store=None,
                    frontier=None) -> "CodeCrawler":
        """Build from a config section; reads ``code_hosts`` when present."""
        crawler = super().from_config(section, fetcher=fetcher,
                                      seen_store=seen_store, frontier=frontier)
        extra = section.get("code_hosts", []) or []
        crawler.code_hosts |= {h.lower() for h in extra}
        return crawler

    def should_follow(self, url: str) -> bool:
        if not super().should_follow(url):
            return False
        parsed = urlparse(url)
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in _CODE_EXTENSIONS):
            return True
        return parsed.netloc.lower() in self.code_hosts

    def extract_content(self, url: str, fetch_result) -> dict:
        """Extract text keeping <pre>/<code> blocks verbatim."""
        body = fetch_result.body or b""
        charset = "utf-8"
        content_type = (getattr(fetch_result, "content_type", "") or "")
        if "charset=" in content_type.lower():
            charset = content_type.lower().split("charset=")[-1].split(";")[0].strip()
        try:
            html = body.decode(charset, errors="replace")
        except (LookupError, ValueError):
            html = body.decode("utf-8", errors="replace")
        return extract_code_text(html, url)
