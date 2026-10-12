"""Disk-quota accounting: walk directories and sum file sizes."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["check_disk_quotas", "dir_size_bytes"]

_BYTES_PER_GB = 1024**3


def dir_size_bytes(path: str | Path) -> int:
    """Return the total size in bytes of all files under ``path``.

    Missing paths count as 0; unreadable files are skipped.
    """
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        try:
            return root.stat().st_size
        except OSError:
            return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                continue
    return total


def check_disk_quotas(paths: dict[str, str | Path], quota_gb: float) -> dict:
    """Summarize disk usage for named paths against a quota.

    Returns ``{"paths": {name: {"bytes": int, "gb": float}},
    "quota_gb": float, "total_gb": float, "over_quota": bool}``.
    """
    quota_gb = float(quota_gb)
    usage: dict[str, dict[str, float | int]] = {}
    for name, path in paths.items():
        size = dir_size_bytes(path)
        usage[name] = {"bytes": size, "gb": size / _BYTES_PER_GB}
    total_bytes = sum(info["bytes"] for info in usage.values())  # type: ignore[arg-type]
    total_gb = total_bytes / _BYTES_PER_GB
    return {
        "paths": usage,
        "quota_gb": quota_gb,
        "total_gb": total_gb,
        "over_quota": total_gb > quota_gb,
    }
