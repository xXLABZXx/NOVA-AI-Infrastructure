"""Sandbox: controlled HTTP client -- the only code allowed to touch the network."""

from .client import FetchResult, SandboxHttpClient
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

__all__ = [
    "SandboxHttpClient",
    "FetchResult",
    "SandboxError",
    "SandboxDisabledError",
    "DomainBlockedError",
    "SSRFBlockedError",
    "DownloadTooLargeError",
    "ContentTypeBlockedError",
    "FetchTimeoutError",
    "FetchFailedError",
    "TokenBucket",
]
