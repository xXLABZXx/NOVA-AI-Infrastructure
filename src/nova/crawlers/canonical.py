"""URL canonicalization for crawl dedup.

Two URLs that differ only by tracking parameters, fragment, case or
percent-encoding noise refer to the same resource. Canonicalizing *before*
enqueueing keeps the frontier (and the seen store) free of duplicates and
prevents the same page from being fetched twice under different URLs.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

__all__ = ["canonicalize_url", "url_fingerprint"]

#: Query parameter names that carry tracking / analytics state rather than
#: content identity. Matched case-insensitively; ``utm_*`` is a prefix rule.
_TRACKING_PARAMS = frozenset(
    {
        "fbclid", "gclid", "gclsrc", "dclid", "wbraid", "gbraid",
        "msclkid", "mc_cid", "mc_eid", "_ga", "_gl", "yclid",
        "ysmad", "vero_conv", "vero_id", "pk_campaign", "pk_kwd",
        "piwik_campaign", "piwik_kwd", "matomo_campaign", "matomo_kwd",
        "sc_campaign", "sc_channel", "sc_content", "sc_medium",
        "sc_outcome", "sc_regionid", "sc_trk", "s_kwcid",
        "ircid", "irgwc", "usg", "ref", "ref_", "source",
        "igshid", "ttclid", "twclid", "li_fat_id",
    }
)
_TRACKING_PREFIXES = ("utm_", "oly_", "sc_")

_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_PCT_ENCODED = re.compile(r"%([0-9A-Fa-f]{2})")


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    if lowered in _TRACKING_PARAMS:
        return True
    return any(lowered.startswith(prefix) for prefix in _TRACKING_PREFIXES)


def _decode_unreserved(path: str) -> str:
    """Decode %XX sequences that represent URI unreserved characters.

    Encodings of reserved / non-ASCII bytes are left untouched so the URL
    keeps its meaning.
    """

    def _replace(match: re.Match) -> str:
        char = chr(int(match.group(1), 16))
        return char if char in _UNRESERVED else match.group(0).upper()

    return _PCT_ENCODED.sub(_replace, path)


def canonicalize_url(url: str) -> str:
    """Return the canonical form of ``url`` used for dedup and frontier keys.

    - lowercases scheme and host, drops default ports (80/443)
    - strips the fragment
    - removes tracking query parameters (``utm_*``, ``fbclid``, ...)
    - sorts the remaining query parameters for a stable order
    - decodes percent-encodings of unreserved characters
    - collapses a trailing ``/`` (``/about/`` -> ``/about``; root stays ``/``)

    Returns ``""`` for unparsable input.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return ""
    host = parsed.hostname or ""
    if not host:
        return ""
    host = host.lower()
    # Drop default ports; keep explicit non-default ones.
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port and not ((scheme == "http" and port == 80)
                     or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = _decode_unreserved(parsed.path or "/")
    # Collapse duplicate slashes and strip trailing slash (except root).
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    if not path:
        path = "/"

    # Filter tracking params, keep the rest sorted for stability.
    kept: list[tuple[str, str]] = []
    try:
        for name, value in parse_qsl(parsed.query, keep_blank_values=True):
            if not _is_tracking_param(name):
                kept.append((name, value))
    except Exception:
        kept = []
    kept.sort(key=lambda pair: (pair[0], pair[1]))
    query = urlencode(kept, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def url_fingerprint(url: str) -> str:
    """SHA-256 hex of the canonical URL: stable dedup key."""
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()
