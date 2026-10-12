"""CLI worker: fetch URLs through the sandbox HTTP client.

Usage:
    python -m nova.sandbox.worker --url URL [--url URL...] [--config configs/config.yaml]

Builds the client from the ``sandbox:`` section of the config, fetches each
URL, and prints one JSON object per line::

    {"url": ..., "status": 200, "bytes": 1234, "sha256": "...", "quarantine_path": ...}
    {"url": ..., "error": "SSRFBlockedError: ..."}

Exit code is 0 unless every fetch failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nova.config import load_config

from .client import SandboxHttpClient
from .errors import SandboxError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch URLs through the NOVA sandbox HTTP client."
    )
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="URL to fetch; repeatable.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configs/config.yaml (defaults to the repo config).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    client = SandboxHttpClient.from_config(config.section("sandbox"))

    failed = 0
    for url in args.url:
        try:
            result = client.fetch(url)
        except SandboxError as exc:
            print(json.dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"}))
            failed += 1
        except Exception as exc:  # noqa: BLE001 - report unexpected failures too
            print(
                json.dumps(
                    {"url": url, "error": f"UnexpectedError: {type(exc).__name__}: {exc}"}
                )
            )
            failed += 1
        else:
            print(
                json.dumps(
                    {
                        "url": result.url,
                        "status": result.status_code,
                        "bytes": len(result.body),
                        "sha256": result.sha256,
                        "quarantine_path": result.quarantine_path,
                    }
                )
            )
    # Exit 0 unless *all* fetches failed.
    return 1 if failed >= len(args.url) else 0


if __name__ == "__main__":
    sys.exit(main())
