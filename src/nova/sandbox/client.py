"""Controlled HTTP client: the ONLY code in NOVA allowed to touch the network.

Every fetch passes, in order: enabled check, scheme check, allow/blocklist,
SSRF validation, rate limiting, download (with retries, size cap and
redirect re-validation), content-type validation, hashing, quarantining,
audit logging, and an optional monitoring event.

Only the standard library is used (``urllib``). Cross-package integrations
(``nova.monitoring``, ``nova.security``) are intentionally *not* imported at
module scope -- those packages are built in parallel -- so this module must
keep working with nothing but ``nova.config`` present.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .errors import (
    ContentTypeBlockedError,
    DomainBlockedError,
    DownloadTooLargeError,
    FetchFailedError,
    FetchTimeoutError,
    SandboxDisabledError,
    SandboxError,
    SSRFBlockedError,
)
from .limits import TokenBucket
from .ssrf import check_host

_CHUNK_SIZE = 65536
_MAX_SINGLE_BACKOFF = 30.0  # seconds; cap for exponential retry backoff


@dataclass
class FetchResult:
    """Outcome of a successful sandboxed fetch."""

    url: str
    status_code: int
    headers: dict = field(default_factory=dict)
    body: bytes = b""
    sha256: str = ""
    content_type: str = ""
    quarantine_path: str | None = None
    elapsed_seconds: float = 0.0


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that re-validates every redirect target.

    The validator raises (DomainBlockedError/SSRFBlockedError) for targets
    that violate the domain policy or SSRF rules, which aborts the redirect.
    """

    def __init__(self, validator: Callable[[str], None]):
        super().__init__()
        self._validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # newurl may be relative; absolutize before validating.
        target = urllib.parse.urljoin(req.full_url, newurl)
        self._validator(target)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _host_matches(host: str, patterns: list[str]) -> bool:
    """True if ``host`` equals or is a subdomain of any pattern."""
    host = host.lower().rstrip(".")
    for pattern in patterns:
        pattern = pattern.lower().rstrip(".")
        if not pattern:
            continue
        if host == pattern or host.endswith("." + pattern):
            return True
    return False


def _redact_url(url: str) -> str:
    """Redact userinfo (``user:password@``) from a URL for logs."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "[unparseable-url]"
    if "@" in parts.netloc:
        _, _, hostinfo = parts.netloc.rpartition("@")
        return parts._replace(netloc="***@" + hostinfo).geturl()
    return url


def _sniff(body: bytes) -> str | None:
    """Magic-byte sniff: 'pdf', 'html', or None when unrecognized/empty."""
    stripped = body.lstrip()
    if not stripped:
        return None
    if stripped.startswith(b"%PDF-"):
        return "pdf"
    head = stripped[:64].lower()
    if head.startswith(b"<html") or head.startswith(b"<!doctype html"):
        return "html"
    return None


class SandboxHttpClient:
    """A policy-enforcing HTTP client.

    Parameters mirror the ``sandbox:`` section of ``configs/config.yaml``;
    use :meth:`from_config` to build one from that section.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
        max_download_bytes: int = 5_242_880,
        timeout_seconds: float = 15,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        requests_per_minute: float = 30,
        allowed_content_types: list[str] | None = None,
        user_agent: str = "NOVA-ResearchBot/0.1",
        quarantine_dir: str | Path | None = None,
        audit_log_path: str | Path | None = None,
        event_logger: Any | None = None,
        allow_private_ips: bool = False,
    ):
        self.enabled = enabled
        self.allowlist = list(allowlist or [])
        self.blocklist = list(blocklist or [])
        self.max_download_bytes = max_download_bytes
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.allowed_content_types = (
            [c.lower() for c in allowed_content_types]
            if allowed_content_types
            else None
        )
        self.user_agent = user_agent
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else None
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self._event_logger = event_logger
        self.allow_private_ips = allow_private_ips
        self._bucket = TokenBucket(requests_per_minute)
        # Passing an *instance* of our HTTPRedirectHandler subclass makes
        # build_opener() skip the default redirect handler, so every redirect
        # goes through _ValidatingRedirectHandler. An empty ProxyHandler
        # disables proxying from environment variables (hermetic, explicit).
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _ValidatingRedirectHandler(self._validate_redirect_target),
        )

    # ------------------------------------------------------------------ config

    @classmethod
    def from_config(cls, section: dict) -> "SandboxHttpClient":
        """Build a client from the ``sandbox:`` section of the config."""
        section = dict(section or {})
        allowed = section.get("allowed_content_types") or None
        return cls(
            enabled=section.get("enabled", True),
            allowlist=list(section.get("allowlist") or []),
            blocklist=list(section.get("blocklist") or []),
            max_download_bytes=int(section.get("max_download_bytes", 5_242_880)),
            timeout_seconds=float(section.get("timeout_seconds", 15)),
            max_retries=int(section.get("max_retries", 3)),
            retry_backoff_seconds=float(
                section.get("retry_backoff_seconds", 2.0)
            ),
            requests_per_minute=float(section.get("requests_per_minute", 30)),
            allowed_content_types=list(allowed) if allowed else None,
            user_agent=section.get("user_agent", "NOVA-ResearchBot/0.1"),
            quarantine_dir=section.get("quarantine_dir"),
            audit_log_path=section.get("audit_log_path"),
            allow_private_ips=bool(section.get("allow_private_ips", False)),
        )

    # ------------------------------------------------------------------- fetch

    def fetch(self, url: str) -> FetchResult:
        """Fetch ``url`` through every sandbox control.

        Raises a :class:`SandboxError` subclass on any policy violation or
        operational failure. Every attempt (success or failure) is audit
        logged when ``audit_log_path`` is set.
        """
        started = time.monotonic()
        status_code = 0
        nbytes = 0
        sha = ""
        content_type = ""
        error: str | None = None
        try:
            result = self._fetch(url, started)
        except Exception as exc:  # noqa: BLE001 - audit must see every failure
            error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            status_code = result.status_code
            nbytes = len(result.body)
            sha = result.sha256
            content_type = result.content_type
            return result
        finally:
            elapsed = time.monotonic() - started
            self._audit(url, status_code, nbytes, sha, content_type, elapsed, error)
            self._emit_event(url, status_code, nbytes, sha, content_type, elapsed, error)

    def _fetch(self, url: str, started: float) -> FetchResult:
        # (1) enabled?
        if not self.enabled:
            raise SandboxDisabledError(
                "sandbox HTTP client is disabled; refusing to fetch"
            )
        # (2) scheme must be http/https.
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            raise DomainBlockedError(
                f"unsupported URL scheme {scheme!r}: only http/https allowed"
            )
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            raise DomainBlockedError(f"URL has no host: {url!r}")
        # (3) blocklist always wins; allowlist gates when non-empty.
        self._check_domain_policy(host, url)
        # (4) SSRF validation (literal IP or every resolved address).
        check_host(host, allow_private_ips=self.allow_private_ips)
        # (5) rate limiting.
        if not self._bucket.acquire(timeout=self.timeout_seconds):
            raise FetchFailedError("rate limit wait timed out")
        # (6)-(10) download, validate, hash, quarantine.
        return self._download(url, host, started)

    # --------------------------------------------------------------- policies

    def _check_domain_policy(self, host: str, url: str) -> None:
        if self.blocklist and _host_matches(host, self.blocklist):
            raise DomainBlockedError(f"host {host!r} is blocklisted")
        if self.allowlist and not _host_matches(host, self.allowlist):
            raise DomainBlockedError(
                f"host {host!r} is not on the allowlist"
            )

    def _validate_redirect_target(self, new_url: str) -> None:
        """Re-check a redirect target against scheme/domain/SSRF policy."""
        parts = urlsplit(new_url)
        if parts.scheme.lower() not in ("http", "https"):
            raise DomainBlockedError(
                f"redirect to unsupported scheme {parts.scheme!r}: {new_url!r}"
            )
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            raise DomainBlockedError(f"redirect target has no host: {new_url!r}")
        self._check_domain_policy(host, new_url)
        check_host(host, allow_private_ips=self.allow_private_ips)

    # --------------------------------------------------------------- download

    def _download(self, url: str, host: str, started: float) -> FetchResult:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        attempts = self.max_retries + 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                    return self._read_response(resp, url, started)
            except urllib.error.HTTPError as exc:
                # Never retry 4xx: the server understood and refused.
                if 400 <= exc.code < 500:
                    raise FetchFailedError(
                        f"HTTP {exc.code} for {url!r}: {exc.reason}"
                    ) from exc
                last_exc = exc
            except (TimeoutError, socket.timeout) as exc:
                last_exc = FetchTimeoutError(
                    f"timed out fetching {url!r} "
                    f"after {self.timeout_seconds}s (attempt {attempt + 1})"
                )
            except urllib.error.URLError as exc:
                # URLError sometimes wraps a timeout in .reason.
                if isinstance(exc.reason, TimeoutError):
                    last_exc = FetchTimeoutError(
                        f"timed out fetching {url!r} (attempt {attempt + 1})"
                    )
                else:
                    last_exc = exc
            except (
                ConnectionError,
                OSError,
                http.client.HTTPException,
                ssl.SSLError,
            ) as exc:
                last_exc = exc
            if attempt < attempts - 1:
                backoff = min(
                    self.retry_backoff_seconds * (2 ** attempt),
                    _MAX_SINGLE_BACKOFF,
                )
                if backoff > 0:
                    time.sleep(backoff)
        if isinstance(last_exc, SandboxError):
            raise last_exc
        raise FetchFailedError(
            f"fetch failed for {url!r} after {attempts} attempt(s): {last_exc}"
        ) from last_exc

    def _read_response(self, resp, url: str, started: float) -> FetchResult:
        # (7) stream the body, aborting past the size cap.
        declared_length = resp.headers.get("Content-Length")
        if declared_length is not None:
            try:
                if int(declared_length) > self.max_download_bytes:
                    raise DownloadTooLargeError(
                        f"Content-Length {declared_length} for {url!r} exceeds "
                        f"max_download_bytes={self.max_download_bytes}"
                    )
            except ValueError:
                pass  # malformed header: fall through to streamed check
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = resp.read(_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > self.max_download_bytes:
                raise DownloadTooLargeError(
                    f"response body for {url!r} exceeds "
                    f"max_download_bytes={self.max_download_bytes}"
                )
            chunks.append(chunk)
        body = b"".join(chunks)
        # (8) content-type validation.
        declared = (
            (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        )
        self._check_content_type(url, declared, body)
        # (9) hash; (10) quarantine.
        digest = hashlib.sha256(body).hexdigest()
        quarantine_path = self._quarantine(body, digest)
        return FetchResult(
            url=url,
            status_code=resp.status,
            headers=dict(resp.headers),
            body=body,
            sha256=digest,
            content_type=declared,
            quarantine_path=quarantine_path,
            elapsed_seconds=time.monotonic() - started,
        )

    def _check_content_type(self, url: str, declared: str, body: bytes) -> None:
        if self.allowed_content_types is not None and (
            declared not in self.allowed_content_types
        ):
            raise ContentTypeBlockedError(
                f"content type {declared!r} for {url!r} "
                f"not in allowed_content_types={self.allowed_content_types}"
            )
        sniffed = _sniff(body)
        if declared == "application/pdf" and sniffed != "pdf":
            raise ContentTypeBlockedError(
                f"declared application/pdf for {url!r} but body does not "
                "start with %PDF-"
            )
        if declared == "text/html" and sniffed not in ("html", None):
            raise ContentTypeBlockedError(
                f"declared text/html for {url!r} but body does not look like HTML"
            )
        # Declared type must agree with what the bytes actually are.
        if declared and sniffed == "pdf" and declared != "application/pdf":
            raise ContentTypeBlockedError(
                f"body for {url!r} looks like PDF but Content-Type is "
                f"{declared!r}"
            )
        if declared and sniffed == "html" and declared != "text/html":
            raise ContentTypeBlockedError(
                f"body for {url!r} looks like HTML but Content-Type is "
                f"{declared!r}"
            )

    def _quarantine(self, body: bytes, digest: str) -> str | None:
        if self.quarantine_dir is None:
            return None
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        path = self.quarantine_dir / f"{digest}.bin"
        path.write_bytes(body)
        return str(path)

    # ------------------------------------------------------------ observability

    def _audit(
        self,
        url: str,
        status_code: int,
        nbytes: int,
        sha: str,
        content_type: str,
        elapsed: float,
        error: str | None,
    ) -> None:
        """Append a JSONL audit record. Never breaks a fetch."""
        if self.audit_log_path is None:
            return
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": _redact_url(url),
            "host": (urlsplit(url).hostname or "") if url else "",
            "status_code": status_code,
            "bytes": nbytes,
            "sha256": sha,
            "content_type": content_type,
            "elapsed_seconds": round(elapsed, 3),
        }
        if error:
            record["error"] = error
        try:
            parent = self.audit_log_path.parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _emit_event(
        self,
        url: str,
        status_code: int,
        nbytes: int,
        sha: str,
        content_type: str,
        elapsed: float,
        error: str | None,
    ) -> None:
        """Notify the optional event logger. Lazy and failure-proof.

        ``nova.monitoring`` is built in parallel, so it is never imported
        here; the injected ``event_logger`` is simply duck-typed for a
        ``.log(name, payload)`` method.
        """
        logger = self._event_logger
        if logger is None:
            return
        try:
            log = getattr(logger, "log", None)
            if callable(log):
                log(
                    "sandbox.fetch",
                    {
                        "url": _redact_url(url),
                        "status_code": status_code,
                        "bytes": nbytes,
                        "sha256": sha,
                        "content_type": content_type,
                        "elapsed_seconds": round(elapsed, 3),
                        "error": error,
                    },
                )
        except Exception:
            pass
