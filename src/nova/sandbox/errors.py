"""Exception hierarchy for the sandbox HTTP client.

Every failure mode of :class:`nova.sandbox.SandboxHttpClient` is a
``SandboxError`` so callers can catch one base class, while the concrete
subclasses let callers distinguish policy blocks (domain/SSRF/content-type)
from operational failures (timeouts, retries exhausted, oversized bodies).
"""


class SandboxError(Exception):
    """Base class for all sandbox errors."""


class SandboxDisabledError(SandboxError):
    """Raised when fetch() is called on a disabled client."""


class DomainBlockedError(SandboxError):
    """Raised when a URL's scheme is unsupported, its host is not allowlisted,
    or its host is blocklisted."""


class SSRFBlockedError(SandboxError):
    """Raised when a URL targets a cloud metadata endpoint or resolves to a
    non-public (private/loopback/link-local/...) IP address."""


class DownloadTooLargeError(SandboxError):
    """Raised when a response body exceeds ``max_download_bytes``."""


class ContentTypeBlockedError(SandboxError):
    """Raised when a response's Content-Type is not in ``allowed_content_types``
    or the declared type disagrees with magic-byte sniffing."""


class FetchTimeoutError(SandboxError):
    """Raised when a fetch exceeds the configured per-attempt timeout."""


class FetchFailedError(SandboxError):
    """Raised when a fetch fails after all retries (connection errors, HTTP 5xx,
    HTTP 4xx without retry, or a rate-limit wait that timed out)."""
