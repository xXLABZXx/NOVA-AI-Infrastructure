"""NOVA dataset factory: continue and improve the base dataset.

Stages: profile -> diversify (teacher rewrites) -> ingest (Common Crawl) ->
mix (dedup + balance + splits + versioned output).
"""

from nova.datasets.base import Record, iter_jsonl_parts, write_jsonl, fingerprint

__all__ = ["Record", "iter_jsonl_parts", "write_jsonl", "fingerprint"]
