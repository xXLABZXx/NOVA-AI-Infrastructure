"""Input validators: URLs, hostnames, records, and integer clamping.

Everything here raises :class:`ValidationError` (a :class:`ValueError`
subclass) on invalid input so callers can catch either.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "ValidationError",
    "validate_url",
    "validate_domain",
    "validate_record",
    "clamp_int",
]


class ValidationError(ValueError):
    """Raised when untrusted input fails validation."""


_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")

_DEFAULT_PORTS = {"http": 80, "https": 443}


def validate_domain(host: str) -> str:
    """Validate a hostname and return it lowercased.

    Accepts an optional single trailing dot (FQDN root). Rejects empty
    labels, labels longer than 63 chars, names longer than 253 chars, and
    anything outside ``[a-z0-9-]`` (including leading/trailing hyphens).
    """
    if not isinstance(host, str):
        raise ValidationError(f"domain must be a string, got {type(host).__name__}")
    name = host.strip().lower()
    if name.endswith("."):
        name = name[:-1]
    if not name:
        raise ValidationError("domain must not be empty")
    if len(name) > 253:
        raise ValidationError(f"domain {name!r} exceeds 253 characters")
    for label in name.split("."):
        if not label:
            raise ValidationError(f"domain {name!r} has an empty label")
        if len(label) > 63:
            raise ValidationError(f"domain label {label!r} exceeds 63 characters")
        if not _LABEL_RE.fullmatch(label):
            raise ValidationError(
                f"domain label {label!r} must use [a-z0-9-] "
                "without leading/trailing hyphens"
            )
    return name


def validate_url(url: str, allowed_schemes: tuple[str, ...] = ("http", "https")) -> str:
    """Validate a URL and return its normalized form.

    Normalization: scheme and host lowercased, default ports dropped,
    fragment removed, embedded credentials rejected. Raises
    :class:`ValidationError` for anything else (bad scheme, missing host,
    invalid port, control characters).
    """
    if not isinstance(url, str):
        raise ValidationError(f"url must be a string, got {type(url).__name__}")
    text = url.strip()
    if not text:
        raise ValidationError("url must not be empty")
    if any(ord(ch) < 0x20 for ch in text):
        raise ValidationError("url contains control characters")
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise ValidationError(f"malformed url {text!r}: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in allowed_schemes:
        raise ValidationError(
            f"url scheme {scheme!r} not in allowed schemes {list(allowed_schemes)}"
        )
    if parts.username or parts.password:
        raise ValidationError("url must not embed credentials")
    host = parts.hostname
    if not host:
        raise ValidationError(f"url {text!r} has no host")
    host = validate_domain(host)
    try:
        port = parts.port
    except ValueError as exc:
        raise ValidationError(f"url {text!r} has an invalid port: {exc}") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValidationError(f"url {text!r} has an invalid port: {port}")
    netloc = host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path, parts.query, ""))


def validate_record(record: dict, required_fields) -> dict:
    """Validate that a mapping record has every required field.

    ``required_fields`` may be a list of field names (presence check) or a
    mapping of field name to expected type (presence + ``isinstance``
    check). Returns the record unchanged on success.
    """
    if not isinstance(record, dict):
        raise ValidationError(
            f"record must be a mapping, got {type(record).__name__}"
        )
    if isinstance(required_fields, dict):
        items = list(required_fields.items())
    else:
        items = [(field, None) for field in required_fields]
    for field, expected in items:
        if field not in record:
            raise ValidationError(f"record missing required field {field!r}")
        if expected is not None and not isinstance(record[field], expected):
            raise ValidationError(
                f"record field {field!r} must be of type "
                f"{getattr(expected, '__name__', expected)}, "
                f"got {type(record[field]).__name__}"
            )
    return record


def clamp_int(value, minimum: int, maximum: int, name: str = "value") -> int:
    """Coerce ``value`` to int and clamp it into ``[minimum, maximum]``.

    Raises :class:`ValidationError` when the value cannot be interpreted
    as an integer (including booleans).
    """
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got bool")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name} must be an integer, got {value!r}"
        ) from exc
    return max(minimum, min(maximum, number))
