"""Size-bounded, strict JSON parsing and serialization.

``safe_loads`` rejects oversized input and non-standard JSON constants
(``NaN``/``Infinity``), which the stdlib ``json`` module accepts by
default. ``safe_load_file`` applies the same size cap before reading a
file from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["safe_loads", "safe_load_file", "safe_dumps"]

#: Default ceiling for a single JSON document (1 MiB).
DEFAULT_MAX_CHARS = 1_000_000


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant not allowed: {value!r}")


def safe_loads(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> Any:
    """Parse JSON strictly.

    Raises :class:`ValueError` if ``text`` exceeds ``max_chars`` or
    contains non-standard constants (``NaN``, ``Infinity``,
    ``-Infinity``). Raises :class:`TypeError` for non-string input.
    """
    if not isinstance(text, str):
        raise TypeError(f"safe_loads expects str, got {type(text).__name__}")
    if len(text) > max_chars:
        raise ValueError(
            f"JSON input of {len(text)} chars exceeds limit of {max_chars}"
        )
    return json.loads(text, parse_constant=_reject_constant)


def safe_load_file(path: str | Path, max_chars: int = DEFAULT_MAX_CHARS) -> Any:
    """Read a JSON file with a size cap and parse it strictly.

    Reads at most ``max_chars + 1`` characters so an oversized file is
    rejected without loading it fully into memory.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, encoding="utf-8") as fh:
        text = fh.read(max_chars + 1)
    if len(text) > max_chars:
        raise ValueError(
            f"JSON file {file_path} exceeds limit of {max_chars} chars"
        )
    return safe_loads(text, max_chars=max_chars)


def safe_dumps(obj: Any) -> str:
    """Serialize to canonical compact JSON (sorted keys, ASCII-safe)."""
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
