"""CLI entry point: ``python -m nova.crawlers.runner --config <cfg> --out <dir> <seeds...>``.

Builds a :class:`GeneralCrawler` on top of the sandbox HTTP client, crawls the
seed URLs, writes ``crawl.jsonl`` into the output directory and prints a
summary. Crawls are resumable: URL state lives in ``<out>/.frontier.db``;
re-running with the same ``--out`` continues where the previous run stopped
instead of refetching completed pages. Permanently failed URLs are written to
``dead_letter.jsonl``. All heavy imports are lazy so the module stays
import-safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_fetcher(sandbox_section: dict):
    """Instantiate the sandbox HTTP client defensively (built in parallel)."""
    from nova.sandbox import SandboxHttpClient

    for kwargs in ({"config": sandbox_section}, {"section": sandbox_section},
                   {}):
        try:
            return SandboxHttpClient(**kwargs)
        except TypeError:
            continue
    # Last resort: try a bare positional call.
    return SandboxHttpClient(sandbox_section)


def write_dead_letter(dead_entries: list[dict], out_path: str | Path) -> Path:
    """Write permanently-failed URLs as JSONL (one ``{url, reason, attempts,
    error}`` per line). Returns the path written."""
    out_path = Path(out_path)
    if str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for entry in dead_entries:
            fh.write(json.dumps({
                "url": entry["url"],
                "reason": entry.get("reason", ""),
                "attempts": entry.get("attempts", 0),
                "error": entry.get("reason", ""),
            }, ensure_ascii=False) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    from nova.config import load_config
    from nova.crawlers.frontier import Frontier
    from nova.crawlers.general import GeneralCrawler
    from nova.crawlers.store import SeenStore

    parser = argparse.ArgumentParser(
        description="NOVA research crawler: crawl seed URLs into JSONL.")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to configs/config.yaml")
    parser.add_argument("--out", default="data/raw",
                        help="Output directory (crawl.jsonl + frontier db)")
    parser.add_argument("seeds", nargs="+", help="Seed URLs to crawl")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    crawler_section = config.section("crawlers")
    sandbox_section = config.section("sandbox")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frontier_db = out_dir / crawler_section.get("frontier_db", ".frontier.db")
    crawl_jsonl = out_dir / "crawl.jsonl"
    dead_letter_jsonl = out_dir / "dead_letter.jsonl"

    resuming = frontier_db.exists() and crawl_jsonl.exists()
    frontier = Frontier(frontier_db)
    seen_store = SeenStore(out_dir / ".seen.db")
    try:
        fetcher = _build_fetcher(sandbox_section)
        crawler = GeneralCrawler.from_config(
            {**crawler_section,
             "user_agent": sandbox_section.get("user_agent",
                                               crawler_section.get("user_agent"))},
            fetcher=fetcher,
            seen_store=seen_store,
            frontier=frontier,
        )
        results = crawler.crawl(args.seeds)
        out_path = crawler.save_results(
            results, crawl_jsonl, mode="a" if resuming else "w")

        dead = frontier.dead_entries()
        if dead:
            write_dead_letter(dead, dead_letter_jsonl)
    finally:
        seen_store.close()
        frontier.close()

    ok = sum(1 for r in results if not r.error)
    failed = len(results) - ok
    print(f"{'resumed' if resuming else 'crawled'}: {len(results)} pages "
          f"({ok} ok, {failed} failed, "
          f"{len(crawler.failed)} recorded failures, "
          f"{crawler.stats.get('dead', 0)} dead-lettered, "
          f"{crawler.stats.get('not_modified', 0)} not-modified)")
    print(f"results: {out_path}")
    print(f"frontier db: {frontier_db}")
    if dead:
        print(f"dead letter: {dead_letter_jsonl} ({len(dead)} URLs)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
