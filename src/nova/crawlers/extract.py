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
    "detect_language_with_confidence",
    "extract_paper_metadata",
    "extract_article",
    "score_quality",
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
    lang, _ = detect_language_with_confidence(text)
    return lang


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """Like :func:`detect_language` but also returns a confidence in [0, 1].

    Confidence is the winner's share of all stopword hits; ``0.0`` accompanies
    ``"unknown"``.
    """
    if not text or len(text) < 50:
        return "unknown", 0.0
    tokens = re.findall(r"[a-zàâäéèêëîïôöùûüçñß]+", text.lower())
    if not tokens:
        return "unknown", 0.0
    scores = {
        lang: sum(1 for tok in tokens if tok in words)
        for lang, words in _STOPWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    if scores[best] == 0 or total == 0:
        return "unknown", 0.0
    return best, round(scores[best] / total, 3)


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


# ---------------------------------------------------------------------------
# Readability-style article extraction and quality scoring
# ---------------------------------------------------------------------------

#: class/id tokens that mark boilerplate containers (nav, ads, chrome...).
_BOILERPLATE_CLASS_RE = re.compile(
    r"comment|sidebar|side-bar|nav(?!-)|footer|header|ad[-_]|ads|advert|"
    r"share|social|related|widget|popup|banner|menu|breadcrumb|subscribe|"
    r"newsletter|cookie|promo|sponsor|login|signin|search[-_]form|"
    r"pagination|pager|toc|tag[-_]cloud|author[-_]bio",
    re.IGNORECASE,
)
#: Structural tags whose whole subtree is boilerplate.
_BOILERPLATE_STRUCTURAL = frozenset({"header", "footer", "nav", "aside"})
#: Tags eligible as article text blocks.
_BLOCK_CANDIDATES = frozenset(
    {"p", "div", "section", "article", "li", "blockquote",
     "h1", "h2", "h3", "h4", "h5", "h6", "td", "pre", "main"}
)
#: Void elements: no end tag is ever emitted for these.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input",
     "link", "meta", "param", "source", "track", "wbr"}
)
class _ArticleExtractor(HTMLParser):
    """Collect candidate text blocks plus document metadata."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.lang = ""
        self.links: list[str] = []
        self.times: list[str] = []
        self.blocks: list[dict] = []
        self._skip_depth = 0
        # Boilerplate scope stack: True entries mean "inside boilerplate".
        # Boilerplate-ness is inherited, so only the top entry matters.
        self._scope_stack: list[bool] = []
        self._in_title = False
        self._in_head = False
        # The currently open candidate block (innermost wins for text).
        self._open: dict | None = None
        self._open_stack: list[dict | None] = []
        self.boilerplate_chars = 0
        self.total_chars = 0

    def _in_boiler(self) -> bool:
        return bool(self._scope_stack and self._scope_stack[-1])

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _is_boilerplate(tag: str, attrs_d: dict) -> bool:
        if tag in _BOILERPLATE_STRUCTURAL:
            return True
        haystack = " ".join(
            str(attrs_d.get(k, "") or "") for k in ("class", "id", "role"))
        return bool(haystack and _BOILERPLATE_CLASS_RE.search(haystack))

    def _close_block(self) -> None:
        block = self._open
        self._open = self._open_stack.pop() if self._open_stack else None
        if block is None:
            return
        text = re.sub(r"\s+", " ", "".join(block["parts"])).strip()
        if len(text) >= 20:  # ignore tiny fragments
            block["text"] = text
            self.blocks.append(block)

    # -- parser callbacks ---------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag not in _VOID_TAGS:
            parent_boiler = self._scope_stack[-1] if self._scope_stack else False
            self._scope_stack.append(
                parent_boiler or self._is_boilerplate(tag, attrs_d))
        if self._in_boiler():
            return
        if tag == "head":
            self._in_head = True
        if tag == "title":
            self._in_title = True
            return
        if tag == "html" and attrs_d.get("lang"):
            self.lang = attrs_d["lang"].split("-")[0].strip().lower()
        if tag == "meta":
            name = (attrs_d.get("name") or attrs_d.get("property") or "").lower()
            content = attrs_d.get("content")
            if name and content:
                self.meta.setdefault(name, content.strip())
        if tag == "a":
            href = attrs_d.get("href")
            if href:
                self.links.append(href.strip())
            # Count upcoming link text separately for density scoring.
            if self._open is not None and not self._skip_depth:
                self._open["in_link"] = self._open.get("in_link", 0) + 1
        if tag == "time":
            dt = attrs_d.get("datetime")
            if dt:
                self.times.append(dt.strip())
        if tag in _BLOCK_CANDIDATES and not self._skip_depth:
            self._open_stack.append(self._open)
            self._open = {
                "tag": tag,
                "class": str(attrs_d.get("class", "")),
                "parts": [],
                "link_chars": 0,
                "in_link": 0,
            }

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag not in _VOID_TAGS and self._scope_stack:
            # HTMLParser is forgiving about mismatched nesting; pop a single
            # scope — well-formed input keeps this exact.
            self._scope_stack.pop()
        if tag == "head":
            self._in_head = False
        if tag == "title":
            self._in_title = False
            return
        if tag == "a" and self._open is not None:
            self._open["in_link"] = max(0, self._open.get("in_link", 0) - 1)
        if tag in _BLOCK_CANDIDATES and self._open is not None \
                and self._open.get("tag") == tag:
            self._close_block()

    def handle_data(self, data: str) -> None:
        if not data or not data.strip():
            return
        self.total_chars += len(data)
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._in_boiler() or self._in_head:
            self.boilerplate_chars += len(data)
            return
        target = self._open
        if target is None:
            # Stray text outside candidate blocks: keep it as its own block
            # so pages without <p> wrappers still yield content. It is
            # appended to ``blocks`` when closed (see ``close()``).
            target = {"tag": "text", "class": "", "parts": [],
                      "link_chars": 0, "in_link": 0}
            self._open = target
            self._open_stack.append(None)
        target["parts"].append(data)
        if target.get("in_link"):
            target["link_chars"] += len(data)

    def close(self) -> None:  # noqa: D102
        while self._open is not None:
            self._close_block()
        super().close()


def _score_block(block: dict) -> float:
    """Readability-style density score: text minus link-text penalty."""
    text = block.get("text", "")
    length = len(text)
    if length == 0:
        return 0.0
    link_ratio = min(1.0, block.get("link_chars", 0) / max(1, length))
    score = length * (1.0 - link_ratio) - length * link_ratio
    if block.get("tag") in ("article", "main"):
        score *= 1.5
    if _BOILERPLATE_CLASS_RE.search(block.get("class", "")):
        score *= 0.25
    return score


def _pick_title(parser: _ArticleExtractor) -> str:
    meta = parser.meta
    for key in ("og:title", "twitter:title"):
        if meta.get(key):
            return meta[key]
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    return title


def _pick_description(parser: _ArticleExtractor) -> str:
    meta = parser.meta
    for key in ("og:description", "twitter:description", "description"):
        if meta.get(key):
            return meta[key]
    return ""


def _pick_publish_date(parser: _ArticleExtractor) -> str:
    meta = parser.meta
    for key in ("article:published_time", "datepublished", "pubdate",
                "publish_date", "dc.date", "sailthru.date"):
        if meta.get(key):
            return meta[key]
    if parser.times:
        return parser.times[0]
    return ""


def _pick_author(parser: _ArticleExtractor) -> str:
    meta = parser.meta
    for key in ("author", "article:author", "twitter:creator", "dc.creator"):
        if meta.get(key):
            return meta[key]
    return ""


def extract_article(html: str | bytes, url: str) -> dict:
    """Extract the main article content from raw HTML.

    Returns ``{title, text, description, publish_date, author, lang,
    lang_confidence, links, url, quality}`` where ``quality`` holds
    ``word_count``, ``text_to_html_ratio``, ``boilerplate_ratio``,
    ``link_text_ratio`` and ``num_blocks``. Boilerplate (nav, ads, chrome)
    is excluded from ``text``; ``links`` still includes every hyperlink.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    parser = _ArticleExtractor()
    parser.feed(html)
    parser.close()

    scored = [(block, _score_block(block)) for block in parser.blocks]
    best = max((score for _, score in scored), default=0.0)
    threshold = max(40.0, best * 0.25)
    kept = [block for block, score in scored if score >= threshold]
    # Preserve document order.
    kept.sort(key=lambda b: parser.blocks.index(b))
    text = "\n\n".join(b["text"] for b in kept).strip()

    lang = parser.lang or "unknown"
    confidence = 0.0
    if not parser.lang:
        lang, confidence = detect_language_with_confidence(text)
    else:
        _, confidence = detect_language_with_confidence(text)
        if lang == "unknown":
            confidence = 0.0

    total = max(1, len(html))
    link_chars = sum(b.get("link_chars", 0) for b in parser.blocks)
    text_chars = sum(len(b.get("text", "")) for b in parser.blocks)
    quality = {
        "word_count": len(text.split()),
        "text_to_html_ratio": round(len(text) / total, 4),
        "boilerplate_ratio": round(
            parser.boilerplate_chars / max(1, parser.total_chars), 4),
        "link_text_ratio": round(link_chars / max(1, text_chars), 4),
        "num_blocks": len(kept),
    }
    quality["score"] = score_quality(
        text, html=html, lang=lang, lang_confidence=confidence,
        boilerplate_ratio=quality["boilerplate_ratio"])["score"]
    return {
        "title": _pick_title(parser),
        "text": text,
        "description": _pick_description(parser),
        "publish_date": _pick_publish_date(parser),
        "author": _pick_author(parser),
        "lang": lang,
        "lang_confidence": confidence,
        "links": list(dict.fromkeys(parser.links)),
        "url": url,
        "quality": quality,
    }


def score_quality(text: str, html: str = "", lang: str = "unknown",
                  lang_confidence: float = 0.0,
                  boilerplate_ratio: float = 0.0) -> dict:
    """Aggregate a 0..1 quality score from extraction metrics.

    Heuristics: minimum word count (100+ words scores fully), text-to-HTML
    ratio, boilerplate penalty and language confidence. Returns the component
    metrics plus ``score``.
    """
    words = len(text.split())
    word_score = min(1.0, words / 100.0)
    html_len = max(1, len(html)) if html else max(1, len(text))
    density = min(1.0, (len(text) / html_len) * 8.0)  # ~12% text is plenty
    boilerplate_penalty = max(0.0, 1.0 - boilerplate_ratio * 2.0)
    lang_score = lang_confidence if lang != "unknown" else 0.2
    score = round(
        0.35 * word_score + 0.25 * density
        + 0.20 * boilerplate_penalty + 0.20 * lang_score,
        3,
    )
    return {
        "score": score,
        "word_count": words,
        "word_score": round(word_score, 3),
        "density": round(density, 3),
        "boilerplate_penalty": round(boilerplate_penalty, 3),
        "lang_score": round(lang_score, 3),
    }
