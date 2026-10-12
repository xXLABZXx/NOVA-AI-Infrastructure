"""CLI entry point: ``python -m nova.crawlers.runner --config <cfg> --out <dir> <seeds...>``.

Builds a :class:`GeneralCrawler` on top of the sandbox HTTP client, crawls the
seed URLs, writes ``crawl.jsonl`` into the output directory and prints a
summary. All heavy imports are lazy so the module stays import-safe.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    from nova.config import load_config
    from nova.crawlers.general import GeneralCrawler
    from nova.crawlers.store import SeenStore

    parser = argparse.ArgumentParser(
        description="NOVA research crawler: crawl seed URLs into JSONL.")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to configs/config.yaml")
    parser.add_argument("--out", default="data/raw",
                        help="Output directory (crawl.jsonl + .seen.db)")
    parser.add_argument("seeds", nargs="+", help="Seed URLs to crawl")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    crawler_section = config.section("crawlers")
    sandbox_section = config.section("sandbox")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_store = SeenStore(out_dir / ".seen.db")
    try:
        fetcher = _build_fetcher(sandbox_section)
        crawler = GeneralCrawler.from_config(
            {**crawler_section,
             "user_agent": sandbox_section.get("user_agent",
                                               crawler_section.get("user_agent"))},
            fetcher=fetcher,
            seen_store=seen_store,
        )
        results = crawler.crawl(args.seeds)
        out_path = crawler.save_results(results, out_dir / "crawl.jsonl")
    finally:
        seen_store.close()

    ok = sum(1 for r in results if not r.error)
    failed = len(results) - ok
    print(f"crawled: {len(results)} pages "
          f"({ok} ok, {failed} failed, "
          f"{len(crawler.failed)} recorded failures)")
    print(f"results: {out_path}")
    print(f"seen db: {seen_store.db_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
