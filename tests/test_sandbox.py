"""Hermetic tests for the sandbox HTTP client.

All network goes through the ``local_http_server`` fixture (127.0.0.1) or
is blocked before any socket opens. No real external network is used.
"""

import hashlib
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from nova.sandbox import (
    ContentTypeBlockedError,
    DomainBlockedError,
    DownloadTooLargeError,
    FetchFailedError,
    FetchTimeoutError,
    SandboxDisabledError,
    SandboxError,
    SandboxHttpClient,
    SSRFBlockedError,
    TokenBucket,
)
from nova.sandbox.ssrf import METADATA_HOSTS, check_host, is_blocked_ip


def make_client(**kwargs):
    """Test client: private IPs allowed (local server), fast retries."""
    kwargs.setdefault("allow_private_ips", True)
    kwargs.setdefault("retry_backoff_seconds", 0.01)
    return SandboxHttpClient(**kwargs)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ------------------------------------------------------------------ success

def test_success(local_http_server, tmp_path):
    body = b"<html><body>hello nova</body></html>"
    local_http_server.routes["/page"] = (
        200,
        {"Content-Type": "text/html; charset=utf-8"},
        body,
    )
    audit = tmp_path / "audit.jsonl"
    qdir = tmp_path / "quarantine"
    client = make_client(quarantine_dir=qdir, audit_log_path=audit)

    result = client.fetch(f"{local_http_server.base_url}/page")

    assert result.status_code == 200
    assert result.body == body
    assert result.content_type == "text/html"  # params stripped
    assert result.sha256 == hashlib.sha256(body).hexdigest()
    assert isinstance(result.headers, dict)
    assert result.elapsed_seconds >= 0

    # Quarantine: <sha256>.bin containing the exact body.
    assert result.quarantine_path is not None
    qpath = Path(result.quarantine_path)
    assert qpath.name == f"{result.sha256}.bin"
    assert qpath.is_file()
    assert qpath.read_bytes() == body

    # Audit JSONL: one record with url/timestamp/status/bytes.
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["url"] == f"{local_http_server.base_url}/page"
    assert record["timestamp"]
    assert record["status_code"] == 200
    assert record["bytes"] == len(body)
    assert record["sha256"] == result.sha256
    assert record["content_type"] == "text/html"
    assert "error" not in record


def test_error_is_audited(local_http_server, tmp_path):
    local_http_server.routes["/nope"] = (
        404, {"Content-Type": "text/plain"}, b"missing"
    )
    audit = tmp_path / "audit.jsonl"
    client = make_client(audit_log_path=audit)
    with pytest.raises(FetchFailedError):
        client.fetch(f"{local_http_server.base_url}/nope")
    record = json.loads(audit.read_text().splitlines()[0])
    assert record["status_code"] == 0
    assert "FetchFailedError" in record["error"]


# ------------------------------------------------------------- domain policy

def test_blocked_domain_not_allowlisted(local_http_server):
    client = make_client(allowlist=["example.com"])
    with pytest.raises(DomainBlockedError):
        client.fetch(f"{local_http_server.base_url}/page")


def test_blocklist_wins_over_allowlist(local_http_server):
    client = make_client(allowlist=["127.0.0.1"], blocklist=["127.0.0.1"])
    with pytest.raises(DomainBlockedError):
        client.fetch(f"{local_http_server.base_url}/page")


def test_subdomain_matches_allowlist(local_http_server):
    # 127.0.0.1 is not a subdomain of anything, so use a permissive check:
    # allowlisting the exact host must permit the fetch.
    client = make_client(allowlist=["127.0.0.1"])
    local_http_server.routes["/ok"] = (
        200, {"Content-Type": "text/plain"}, b"fine"
    )
    result = client.fetch(f"{local_http_server.base_url}/ok")
    assert result.status_code == 200


def test_unsupported_scheme():
    client = make_client()
    with pytest.raises(DomainBlockedError):
        client.fetch("ftp://example.com/file")


# -------------------------------------------------------------------- SSRF

def test_ssrf_metadata_literal_no_network():
    client = SandboxHttpClient()  # default: private IPs NOT allowed
    with pytest.raises(SSRFBlockedError):
        client.fetch("http://169.254.169.254/")


def test_ssrf_loopback_default_blocked():
    client = SandboxHttpClient()
    with pytest.raises(SSRFBlockedError):
        client.fetch("http://127.0.0.1:9/")


def test_ssrf_loopback_allowed_with_flag(local_http_server):
    client = make_client()  # allow_private_ips=True
    local_http_server.routes["/local"] = (
        200, {"Content-Type": "text/plain"}, b"local ok"
    )
    result = client.fetch(f"{local_http_server.base_url}/local")
    assert result.status_code == 200


def test_redirect_target_revalidated(local_http_server):
    # A redirect to a metadata IP must be blocked even when private IPs
    # are allowed for the initial hop.
    local_http_server.routes["/redir"] = (
        302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b""
    )
    client = make_client()
    with pytest.raises(SSRFBlockedError):
        client.fetch(f"{local_http_server.base_url}/redir")


def test_is_blocked_ip():
    for ip in (
        "10.0.0.1", "172.16.5.4", "192.168.1.1",  # private
        "127.0.0.1", "::1",  # loopback
        "169.254.169.254", "fe80::1",  # link-local
        "224.0.0.1", "ff02::1",  # multicast
        "0.0.0.0", "::",  # unspecified
        "240.0.0.1",  # reserved
    ):
        assert is_blocked_ip(ip), ip
    for ip in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"):
        assert not is_blocked_ip(ip), ip
    assert METADATA_HOSTS == {"169.254.169.254", "169.254.169.253", "fd00:ec2::254"}
    with pytest.raises(SSRFBlockedError):
        check_host("metadata.google.internal")


# ------------------------------------------------------------------ limits

def test_size_limit(local_http_server):
    big = b"x" * (2 * 1024 * 1024)
    local_http_server.routes["/big"] = (
        200, {"Content-Type": "application/octet-stream"}, big
    )
    client = make_client(max_download_bytes=1024)
    with pytest.raises(DownloadTooLargeError):
        client.fetch(f"{local_http_server.base_url}/big")


def test_token_bucket():
    bucket = TokenBucket(rate_per_minute=60, capacity=1)
    assert bucket.acquire() is True
    assert bucket.acquire(timeout=0) is False


# ----------------------------------------------------------------- disabled

def test_disabled():
    client = SandboxHttpClient(enabled=False)
    with pytest.raises(SandboxDisabledError):
        client.fetch("http://example.com/")


# ------------------------------------------------------------------ retries

def test_retries_exhausted_raises_fetch_failed():
    port = free_port()  # nothing listening: connection refused
    client = SandboxHttpClient(
        allow_private_ips=True, max_retries=2, retry_backoff_seconds=0.01
    )
    with pytest.raises(FetchFailedError):
        client.fetch(f"http://127.0.0.1:{port}/")


def test_no_retry_on_4xx(local_http_server):
    local_http_server.routes["/gone"] = (
        404, {"Content-Type": "text/plain"}, b"gone"
    )
    client = make_client(max_retries=3)
    with pytest.raises(FetchFailedError):
        client.fetch(f"{local_http_server.base_url}/gone")


# ------------------------------------------------------------ content types

def test_content_type_sniff_mismatch(local_http_server):
    local_http_server.routes["/pdf"] = (
        200, {"Content-Type": "text/html"}, b"%PDF-1.4 fake pdf body"
    )
    client = make_client(allowed_content_types=["application/pdf"])
    with pytest.raises(ContentTypeBlockedError):
        client.fetch(f"{local_http_server.base_url}/pdf")


def test_content_type_not_allowed(local_http_server):
    local_http_server.routes["/json"] = (
        200, {"Content-Type": "application/json"}, b'{"a": 1}'
    )
    client = make_client(allowed_content_types=["text/html"])
    with pytest.raises(ContentTypeBlockedError):
        client.fetch(f"{local_http_server.base_url}/json")


def test_declared_pdf_but_not_pdf_bytes(local_http_server):
    local_http_server.routes["/fakepdf"] = (
        200, {"Content-Type": "application/pdf"}, b"not a pdf at all"
    )
    client = make_client()
    with pytest.raises(ContentTypeBlockedError):
        client.fetch(f"{local_http_server.base_url}/fakepdf")


# --------------------------------------------------------------- hierarchy

def test_exception_hierarchy():
    for cls in (
        SandboxDisabledError,
        DomainBlockedError,
        SSRFBlockedError,
        DownloadTooLargeError,
        ContentTypeBlockedError,
        FetchTimeoutError,
        FetchFailedError,
    ):
        assert issubclass(cls, SandboxError)


# ------------------------------------------------------------------ config

def test_from_config():
    section = {
        "enabled": True,
        "allowlist": ["example.com"],
        "blocklist": [],
        "max_download_bytes": 1024,
        "timeout_seconds": 5,
        "max_retries": 1,
        "retry_backoff_seconds": 0.5,
        "requests_per_minute": 10,
        "allowed_content_types": ["text/html"],
        "user_agent": "TestBot/1.0",
    }
    client = SandboxHttpClient.from_config(section)
    assert client.enabled is True
    assert client.allowlist == ["example.com"]
    assert client.max_download_bytes == 1024
    assert client.timeout_seconds == 5
    assert client.max_retries == 1
    assert client.retry_backoff_seconds == 0.5
    assert client.allowed_content_types == ["text/html"]
    assert client.user_agent == "TestBot/1.0"


def test_from_config_real_config_file():
    from nova.config import load_config

    config = load_config()  # repo configs/config.yaml
    client = SandboxHttpClient.from_config(config.section("sandbox"))
    assert client.enabled is True
    assert "arxiv.org" in client.allowlist
    assert client.max_download_bytes == 5242880
    assert "text/html" in (client.allowed_content_types or [])


# ------------------------------------------- extra headers + 304 handling


class _FakeResponse:
    """Minimal urllib response stand-in for opener stubbing."""

    def __init__(self, status, headers, body):
        self.status = status
        self.headers = dict(headers)
        self._body = body

    def read(self, n=-1):
        if not self._body:
            return b""
        chunk, self._body = self._body[:n], self._body[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _CapturingOpener:
    """Stand-in for the urllib opener: records requests, replays behavior."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.requests = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        return self.behavior(req)


def test_extra_headers_reach_the_request(tmp_path):
    seen = {}

    def behavior(req):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeResponse(
            200, {"Content-Type": "text/html"}, b"<html>ok</html>"
        )

    client = make_client()
    client._opener = _CapturingOpener(behavior)
    result = client.fetch(
        "http://127.0.0.1/page",
        headers={"If-None-Match": '"abc123"',
                 "If-Modified-Since": "Mon, 12 Oct 2026 00:00:00 GMT"},
    )
    assert result.status_code == 200
    assert seen["headers"]["if-none-match"] == '"abc123"'
    assert seen["headers"]["if-modified-since"] == \
        "Mon, 12 Oct 2026 00:00:00 GMT"
    # The configured default User-Agent is still sent alongside.
    assert seen["headers"]["user-agent"] == client.user_agent


def test_header_user_agent_override(tmp_path):
    seen = {}

    def behavior(req):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeResponse(
            200, {"Content-Type": "text/html"}, b"<html>ok</html>"
        )

    client = make_client(user_agent="DefaultBot/1.0")
    client._opener = _CapturingOpener(behavior)
    client.fetch("http://127.0.0.1/page",
                 headers={"User-Agent": "NOVA-Bot/0.2"})
    # Only one User-Agent header may go out (no duplicates).
    uas = [v for k, v in seen["headers"].items() if k == "user-agent"]
    assert uas == ["NOVA-Bot/0.2"]


def test_304_returns_result_without_retry(tmp_path):
    calls = []

    def behavior(req):
        calls.append(req)
        raise urllib.error.HTTPError(
            req.full_url, 304, "Not Modified", {"ETag": '"abc123"'}, None
        )

    audit = tmp_path / "audit.jsonl"
    qdir = tmp_path / "quarantine"
    client = make_client(quarantine_dir=qdir, audit_log_path=audit)
    client._opener = _CapturingOpener(behavior)

    result = client.fetch("http://127.0.0.1/page",
                          headers={"If-None-Match": '"abc123"'})

    assert result.status_code == 304
    assert result.body == b""
    assert result.quarantine_path is None  # nothing new to quarantine
    assert len(calls) == 1  # 304 is final: never retried

    records = [json.loads(line)
               for line in audit.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status_code"] == 304
    assert "error" not in records[0]


def test_304_from_real_server(local_http_server):
    local_http_server.routes["/cond"] = (
        304, {"ETag": '"abc123"'}, b""
    )
    client = make_client()
    result = client.fetch(f"{local_http_server.base_url}/cond",
                          headers={"If-None-Match": '"abc123"'})
    assert result.status_code == 304
    assert result.body == b""


def test_disallowed_header_rejected():
    client = make_client()
    with pytest.raises(SandboxError):
        client.fetch("http://127.0.0.1/page", headers={"Host": "evil.com"})
    with pytest.raises(SandboxError):
        client.fetch("http://127.0.0.1/page",
                     headers={"Cookie": "session=abc"})
    with pytest.raises(SandboxError):
        client.fetch("http://127.0.0.1/page",
                     headers={"Authorization": "Bearer x"})


def test_header_injection_rejected():
    client = make_client()
    with pytest.raises(SandboxError):
        client.fetch("http://127.0.0.1/page",
                     headers={"If-None-Match": '"a"\r\nEvil: 1'})
    with pytest.raises(SandboxError):
        client.fetch("http://127.0.0.1/page",
                     headers={"If-None-Match\r\nEvil": "x"})
    with pytest.raises(SandboxError):
        client.fetch("http://127.0.0.1/page",
                     headers={"If-None-Match": 123})  # not a string
