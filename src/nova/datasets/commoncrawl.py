"""Common Crawl bulk ingestion (no AWS credentials needed).

Uses the free HTTP endpoints:
  - crawl list:  https://index.commoncrawl.org/collinfo.json
  - index API:   https://index.commoncrawl.org/<CRAWL>-index?url=<domain>/*&output=json
  - WET data:    https://data.commoncrawl.org/<filename>  (HTTP Range requests)

WET (extracted text) records are parsed with a minimal stdlib WARC parser,
language-filtered, and emitted as ``kind="text"`` records. A checkpoint file
records completed (filename, offset) ranges so interrupted runs resume.

CLI: ``python -m nova.datasets.commoncrawl --crawl CC-MAIN-2026-39
--domains example.com --out cc.jsonl --max-records 10000``
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

INDEX_BASE = "https://index.commoncrawl.org"
DATA_BASE = "https://data.commoncrawl.org"
UA = "NOVA-DatasetFactory/0.1 (+https://github.com/xXLABZXx/NOVA-AI-Infrastructure; research ingestion)"


def _http_get(url: str, *, timeout: int = 60, retries: int = 3,
              headers: dict | None = None) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def list_crawls() -> list[dict]:
    """Return available crawls from collinfo.json."""
    return json.loads(_http_get(f"{INDEX_BASE}/collinfo.json").decode("utf-8"))


def query_index(crawl: str, domain: str, *, max_pages: int = 50) -> list[dict]:
    """Query the columnar index for ``<domain>/*``. Returns record dicts."""
    records: list[dict] = []
    page = 0
    while page < max_pages:
        qs = urllib.parse.urlencode({
            "url": f"{domain}/*",
            "output": "json",
            "filter": "status:200",
            "page": page,
        })
        url = f"{INDEX_BASE}/{crawl}-index?{qs}"
        try:
            raw = _http_get(url).decode("utf-8", errors="replace")
        except RuntimeError:
            break
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            break
        for ln in lines:
            try:
                records.append(json.loads(ln))
            except ValueError:
                continue
        if len(lines) < 15000:  # last page (default pageSize)
            break
        page += 1
        time.sleep(1.0)  # be polite to the index API
    return records


def fetch_range(filename: str, offset: int, length: int,
               *, retries: int = 3) -> bytes:
    """Download a byte range from data.commoncrawl.org (decompressed)."""
    url = f"{DATA_BASE}/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, **headers})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            buf = io.BytesIO(data)
            with gzip.GzipFile(fileobj=buf) as gz:
                return gz.read()
        except Exception as exc:
            last = exc
            time.sleep(min(60.0, 3.0 * (attempt + 1)))
    raise RuntimeError(f"range fetch failed: {filename}@{offset} ({last})")


def parse_wet_records(payload: bytes) -> list[dict]:
    """Minimal WET/WARC parser -> [{uri, date, text}]."""
    records: list[dict] = []
    pos = 0
    n = len(payload)
    while True:
        start = payload.find(b"WARC/1.0\r\n", pos)
        if start < 0:
            break
        h_end = payload.find(b"\r\n\r\n", start)
        if h_end < 0:
            break
        headers: dict[str, str] = {}
        for line in payload[start:h_end].decode("utf-8", errors="replace").split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            length = 0
        body_start = h_end + 4
        body = payload[body_start:body_start + length]
        text = body.decode("utf-8", errors="replace").strip()
        if headers.get("warc-type") == "conversion" and text:
            records.append({
                "uri": headers.get("warc-target-uri", ""),
                "date": headers.get("warc-date", ""),
                "text": text,
            })
        pos = body_start + length
        if pos >= n:
            break
    return records


def _detect_language(text: str) -> str:
    try:
        from nova.crawlers.extract import detect_language
        return detect_language(text)
    except Exception:
        return "unknown"


def ingest(
    crawl: str,
    domains: list[str],
    out_path: str,
    *,
    max_records: int = 0,
    language: str = "en",
    min_chars: int = 200,
    checkpoint_path: str = "",
    polite_delay: float = 0.5,
) -> dict:
    """Stream WET records -> JSONL text records. Returns stats."""
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    done: set[str] = set()
    if ckpt and ckpt.exists():
        try:
            done = set(json.loads(ckpt.read_text(encoding="utf-8")).get("done", []))
        except ValueError:
            done = set()

    from nova.datasets.base import write_jsonl

    emitted = skipped_lang = skipped_short = skipped_dup_range = 0
    buf: list[dict] = []

    def flush():
        nonlocal buf
        if buf:
            write_jsonl(out_path, iter(buf), append=True)
            buf = []

    def save_ckpt():
        if ckpt:
            ckpt.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")

    try:
        for domain in domains:
            for entry in query_index(crawl, domain):
                key = f"{entry.get('filename')}@{entry.get('offset')}"
                if key in done:
                    skipped_dup_range += 1
                    continue
                if max_records and emitted >= max_records:
                    break
                try:
                    payload = fetch_range(entry["filename"],
                                          int(entry["offset"]), int(entry["length"]))
                except RuntimeError:
                    continue
                for rec in parse_wet_records(payload):
                    text = rec["text"]
                    if len(text) < min_chars:
                        skipped_short += 1
                        continue
                    if language and _detect_language(text) != language:
                        skipped_lang += 1
                        continue
                    if max_records and emitted >= max_records:
                        break
                    buf.append({
                        "id": f"cc-{crawl}-{entry.get('digest', '')[:16]}",
                        "kind": "text",
                        "category": "web",
                        "text": text[:20000],
                        "source": rec["uri"],
                        "synthetic": False,
                        "provenance": {
                            "source": "commoncrawl",
                            "crawl": crawl,
                            "warc_date": rec["date"],
                        },
                    })
                    emitted += 1
                    if len(buf) >= 200:
                        flush()
                done.add(key)
                if len(done) % 25 == 0:
                    save_ckpt()
                time.sleep(polite_delay)
            if max_records and emitted >= max_records:
                break
    finally:
        flush()
        save_ckpt()
    return {
        "emitted": emitted,
        "skipped_language": skipped_lang,
        "skipped_too_short": skipped_short,
        "skipped_resumed_ranges": skipped_dup_range,
        "out_path": out_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Common Crawl WET text (free HTTP, no AWS key).")
    parser.add_argument("--crawl", default="",
                        help="Crawl id like CC-MAIN-2026-39 (default: latest)")
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--checkpoint", default="cc-checkpoint.json")
    args = parser.parse_args(argv)

    crawl = args.crawl
    if not crawl:
        crawls = list_crawls()
        crawl = crawls[0]["id"]
        print(f"using latest crawl: {crawl}")
    stats = ingest(
        crawl, args.domains, args.out,
        max_records=args.max_records,
        language=args.language,
        min_chars=args.min_chars,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
