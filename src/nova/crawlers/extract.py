"""HTML text extraction and language detection (stdlib only).

``extract_text`` pulls a page's title, visible text, language hint and outgoing
links out of raw HTML. ``extract_code_text`` does the same but keeps
``<pre>``/``<code>`` blocks verbatim for source-code pages.
``detect_language`` is a small stopword-based heuristic for en/fr/de/es.
``extract_paper_metadata`` heuristically finds a paper title/abstract.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

__all__ = [
    "extract_text",
    "extract_code_text",
    "detect_language",
    "extract_paper_metadata",
]

#: Tags whose text is never part of the visible page content.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
#: Boilerplate-ish containers whose *text* is dropped (links are still kept).
_BOILERPLATE_TAGS = frozenset({"nav", "footer", "aside"})
#: Tags that force a line break in the extracted text.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "main", "header",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "ul", "ol", "tr", "table", "br", "hr",
        "blockquote", "figure", "figcaption", "pre",
    }
)


class _TextExtractor(HTMLParser):
    """Single-pass HTML -> (title, text chunks, lang, links)."""

    def __init__(self, *, keep_code_verbatim: bool = False):
        super().__init__(convert_charrefs=True)
        self.keep_code_verbatim = keep_code_verbatim
        self.title_parts: list[str] = []
        self.lang = ""
        self.links: list[str] = []
        # Chunks are (kind, text) with kind in {"text", "code"}.
        self.chunks: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._boilerplate_depth = 0
        self._code_depth = 0
        self._in_title = False
        self._in_head = False

    # -- tag handling -----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in _BOILERPLATE_TAGS:
            self._boilerplate_depth += 1
        if tag == "head":
            self._in_head = True
        if tag == "title":
            self._in_title = True
            return
        if tag == "html" and attrs_d.get("lang"):
            self.lang = attrs_d["lang"].split("-")[0].strip().lower()
        if tag == "a":
            href = attrs_d.get("href")
            if href:
                self.links.append(href.strip())
        if tag in ("pre", "code") and self.keep_code_verbatim:
            self._code_depth += 1
        if tag in _BLOCK_TAGS:
            self.chunks.append(("text", "\n"))

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in _BOILERPLATE_TAGS:
            self._boilerplate_depth = max(0, self._boilerplate_depth - 1)
        if tag == "head":
            self._in_head = False
        if tag == "title":
            self._in_title = False
        if tag in ("pre", "code") and self._code_depth:
            self._code_depth -= 1
        if tag in _BLOCK_TAGS:
            self.chunks.append(("text", "\n"))

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._boilerplate_depth or self._in_head:
            return
        if self._code_depth and self.keep_code_verbatim:
            if data.strip():
                self.chunks.append(("code", data))
            return
        text = data.strip()
        if text:
            self.chunks.append(("text", text + " "))

    # -- assembly ---------------------------------------------------------
    def get_text(self) -> str:
        parts: list[str] = []
        for kind, text in self.chunks:
            if kind == "code":
                # Keep code blocks verbatim (trailing whitespace trimmed only).
                parts.append("\n" + text.rstrip() + "\n")
            else:
                parts.append(re.sub(r"\s+", " ", text))
        merged = "".join(parts)
        merged = re.sub(r"[ \t]+\n", "\n", merged)
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        return merged.strip()

    def get_title(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.title_parts)).strip()


def _run_extractor(html: str, url: str, *, keep_code_verbatim: bool) -> dict:
    parser = _TextExtractor(keep_code_verbatim=keep_code_verbatim)
    parser.feed(html)
    parser.close()
    text = parser.get_text()
    lang = parser.lang or detect_language(text)
    return {
        "title": parser.get_title(),
        "text": text,
        "lang": lang,
        "links": parser.links,
        "url": url,
    }


def extract_text(html: str | bytes, url: str) -> dict:
    """Extract ``{title, text, lang, links, url}`` from raw HTML.

    Skips ``script``/``style``/``nav``/``footer``-ish content, resolves nothing
    (links are returned raw; the caller resolves them with ``urljoin``).
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    return _run_extractor(html, url, keep_code_verbatim=False)


def extract_code_text(html: str | bytes, url: str) -> dict:
    """Like :func:`extract_text` but keeps ``<pre>``/``<code>`` blocks verbatim.

    Used by the code crawler so indentation and line breaks in source listings
    survive extraction.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    return _run_extractor(html, url, keep_code_verbatim=True)


_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {"the", "and", "of", "to", "in", "is", "that", "it", "was", "for",
         "on", "are", "as", "with", "his", "they", "at", "be", "this",
         "have", "from", "or", "an", "a", "by", "not", "but"}
    ),
    "fr": frozenset(
        {"le", "la", "les", "de", "des", "et", "est", "en", "un", "une",
         "que", "pour", "dans", "du", "au", "il", "elle", "nous", "vous",
         "pas", "ce", "cette", "sont", "plus", "sur"}
    ),
    "de": frozenset(
        {"der", "die", "das", "und", "ist", "in", "den", "von", "zu",
         "mit", "sich", "des", "auf", "für", "nicht", "ein", "eine",
         "als", "auch", "dem", "einer"}
    ),
    "es": frozenset(
        {"el", "la", "los", "las", "de", "que", "y", "en", "un", "una",
         "es", "por", "con", "para", "como", "del", "al", "se", "su",
         "lo", "más", "pero", "sus"}
    ),
}


def detect_language(text: str) -> str:
    """Heuristic language id over ``{en, fr, de, es}`` or ``"unknown"``.

    Scores each language by stopword hits. Returns ``"unknown"`` when the
    text is too short (< 50 chars) or no stopwords match at all.
    """
    if not text or len(text) < 50:
        return "unknown"
    tokens = re.findall(r"[a-zàâäéèêëîïôöùûüçñß]+", text.lower())
    if not tokens:
        return "unknown"
    scores = {
        lang: sum(1 for tok in tokens if tok in words)
        for lang, words in _STOPWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "unknown"
    return best


def extract_paper_metadata(html: str | bytes) -> dict:
    """Heuristically extract ``{title, abstract}`` from a paper landing page.

    Tries, in order: ``<meta name="citation_title">`` / first ``<h1>`` for the
    title, and ``<meta name="citation_abstract">`` / ``<meta name="description">``
    / ``<blockquote class="abstract">`` for the abstract. Missing values are
    empty strings.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")

    meta: dict[str, str] = {}
    for match in re.finditer(
        r'<meta\s[^>]*?name=["\']([^"\']+)["\'][^>]*?content=["\']([^"\']*)["\']',
        html,
        flags=re.IGNORECASE,
    ):
        meta[match.group(1).lower()] = match.group(2).strip()
    # content-before-name ordering variant
    for match in re.finditer(
        r'<meta\s[^>]*?content=["\']([^"\']*)["\'][^>]*?name=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    ):
        meta.setdefault(match.group(2).lower(), match.group(1).strip())

    title = meta.get("citation_title", "")
    if not title:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
        if h1:
            title = re.sub(r"<[^>]+>", "", h1.group(1))
            title = re.sub(r"\s+", " ", title).strip()

    abstract = meta.get("citation_abstract", "") or meta.get("description", "")
    if not abstract:
        bq = re.search(
            r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</blockquote>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if bq:
            abstract = re.sub(r"<[^>]+>", "", bq.group(1))
            abstract = re.sub(r"\s+", " ", abstract).strip()
            abstract = re.sub(r"^(abstract:?\s*)", "", abstract, flags=re.IGNORECASE)

    return {"title": title, "abstract": abstract}
