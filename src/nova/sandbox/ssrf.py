"""Server-Side Request Forgery (SSRF) protection.

Every outbound fetch target is validated before any socket is opened:

* literal IP hosts are checked directly;
* DNS hostnames are resolved with :func:`socket.getaddrinfo` and **every**
  resolved address must be public;
* well-known cloud metadata endpoints are blocked explicitly.

No reverse-DNS lookups are ever performed.
"""

from __future__ import annotations

import ipaddress
import socket

from .errors import SSRFBlockedError

#: Literal IPs of well-known cloud instance-metadata endpoints.
METADATA_HOSTS = {"169.254.169.254", "169.254.169.253", "fd00:ec2::254"}

#: Hostnames of well-known cloud instance-metadata endpoints.
METADATA_HOSTNAMES = {"metadata.google.internal"}


def is_blocked_ip(ip_str: str) -> bool:
    """Return True if ``ip_str`` must never be a fetch target.

    Blocks private, loopback, link-local, multicast, reserved and
    unspecified addresses, for both IPv4 and IPv6. Unparsable input is
    treated as blocked (fail closed).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _strip_brackets(host: str) -> str:
    """Remove IPv6 literal brackets, e.g. ``[::1]`` -> ``::1``."""
    if host.startswith("[") and host.endswith("]") and len(host) > 2:
        return host[1:-1]
    return host


def check_host(host: str, allow_private_ips: bool = False) -> None:
    """Validate a fetch target host, raising :class:`SSRFBlockedError`.

    * Metadata hostnames (e.g. ``metadata.google.internal``) are rejected.
    * Literal IPs are checked directly against :data:`METADATA_HOSTS` and
      :func:`is_blocked_ip`.
    * Other names are resolved with :func:`socket.getaddrinfo`; **every**
      resolved address is checked. A resolution failure is itself a block.

    ``allow_private_ips=True`` (tests / controlled environments) skips the
    non-public-IP check, but metadata endpoints stay blocked.
    """
    if not host or not host.strip():
        raise SSRFBlockedError("blocked: empty host")
    host = _strip_brackets(host.strip())

    if host.lower() in METADATA_HOSTNAMES:
        raise SSRFBlockedError(
            f"blocked: {host!r} is a cloud metadata endpoint"
        )

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        candidates = [str(literal)]
    else:
        try:
            infos = socket.getaddrinfo(
                host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise SSRFBlockedError(
                f"blocked: host {host!r} could not be resolved ({exc})"
            ) from exc
        candidates = []
        for info in infos:
            addr = info[4][0]
            if addr not in candidates:
                candidates.append(addr)
        if not candidates:
            raise SSRFBlockedError(
                f"blocked: host {host!r} resolved to no addresses"
            )

    for addr in candidates:
        if addr in METADATA_HOSTS or _strip_brackets(addr) in METADATA_HOSTS:
            raise SSRFBlockedError(
                f"blocked: host {host!r} targets cloud metadata IP {addr}"
            )
        if allow_private_ips:
            continue
        if is_blocked_ip(addr):
            raise SSRFBlockedError(
                f"blocked: host {host!r} resolves to non-public IP {addr} "
                "(private/loopback/link-local/multicast/reserved/unspecified)"
            )
