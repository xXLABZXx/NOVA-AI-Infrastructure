"""Shared pytest fixtures: synthetic data, temp workspaces, local HTTP server.

All tests must be hermetic: no real network, no paid APIs, no torch required.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_doc(doc_id, url, text, lang="en", title=None, source="test"):
    import hashlib

    return {
        "id": doc_id,
        "url": url,
        "title": title or f"Title {doc_id}",
        "text": text,
        "lang": lang,
        "source": source,
        "retrieved_at": "2026-10-12T12:00:00Z",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


@pytest.fixture
def synthetic_docs():
    """A small synthetic corpus used by ingestion/dedup/curation/training tests."""
    return [
        make_doc(
            "doc-1",
            "https://example.com/a",
            "Machine learning is a field of artificial intelligence. " * 20,
        ),
        make_doc(
            "doc-2",
            "https://example.com/b",
            "Deep learning uses neural networks with many layers. " * 20,
        ),
        # Exact duplicate of doc-1 under a different id/url.
        make_doc(
            "doc-3",
            "https://example.com/c",
            "Machine learning is a field of artificial intelligence. " * 20,
        ),
        # Near-duplicate of doc-2 (one sentence changed).
        make_doc(
            "doc-4",
            "https://example.com/d",
            ("Deep learning uses neural networks with many layers. " * 19)
            + "Deep learning uses neural networks with several layers. ",
        ),
        make_doc(
            "doc-5",
            "https://example.com/e",
            "x",  # too short / low quality
        ),
    ]


@pytest.fixture
def synthetic_jsonl(tmp_path, synthetic_docs):
    path = tmp_path / "synthetic.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for doc in synthetic_docs:
            fh.write(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def workspace_dirs(tmp_path):
    """Standard data/log/quarantine layout inside a temp dir."""
    dirs = {}
    for name in ("raw", "cleaned", "approved", "rejected", "logs", "quarantine"):
        p = tmp_path / name
        p.mkdir()
        dirs[name] = p
    return dirs


class _QuietHandler(BaseHTTPRequestHandler):
    """Minimal controllable HTTP server for sandbox/crawler tests."""

    routes = {}  # path -> (status, headers, body)

    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):
        status, headers, body = self.routes.get(
            self.path, (404, {"Content-Type": "text/plain"}, b"not found")
        )
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def local_http_server():
    """A real local HTTP server on 127.0.0.1 with configurable routes.

    Usage: server.routes["/page"] = (200, {"Content-Type": "text/html"}, b"<html>...")
             url = f"{server.base_url}/page"
    """
    _QuietHandler.routes = {}
    server = HTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.base_url = f"http://127.0.0.1:{server.server_port}"
    # Expose the handler's route table on the server, as the docstring promises:
    #   server.routes["/page"] = (200, {"Content-Type": "text/html"}, b"<html>...")
    server.routes = _QuietHandler.routes
    yield server
    server.shutdown()
    thread.join()
