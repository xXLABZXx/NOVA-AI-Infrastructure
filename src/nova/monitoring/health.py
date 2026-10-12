"""Health summary: module flags, disk quotas, and overall status.

``status`` is ``"degraded"`` when disk usage exceeds the configured
quota, ``"ok"`` otherwise. Run directly for a CLI report::

    python -m nova.monitoring.health
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.monitoring.quotas import check_disk_quotas

__all__ = ["health_summary"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_paths() -> dict[str, Path]:
    """Repo-root-relative data/logs/quarantine directories."""
    from nova.config import default_config_path  # lazy: keeps import light

    root = default_config_path().parent.parent
    return {
        "data": root / "data",
        "logs": root / "logs",
        "quarantine": root / "quarantine",
    }


def health_summary(
    config: dict | None = None,
    paths: dict[str, str | Path] | None = None,
) -> dict:
    """Build a health report dict.

    ``config`` may be a :class:`~nova.config.NovaConfig` or a plain
    mapping; when omitted, the default config file is loaded. ``paths``
    maps names to directories for quota accounting; defaults to the
    repo's data/logs/quarantine directories.
    """
    from nova.config import KNOWN_MODULES, load_config  # lazy: avoids import cycle

    cfg: Any = config if config is not None else load_config()
    modules_cfg = cfg.get("modules", {}) if isinstance(cfg, dict) else {}
    modules = {name: bool(modules_cfg.get(name, False)) for name in KNOWN_MODULES}

    monitoring_cfg = cfg.get("monitoring", {}) if isinstance(cfg, dict) else {}
    try:
        quota_gb = float(monitoring_cfg.get("disk_quota_gb", 50))
    except (TypeError, ValueError):
        quota_gb = 50.0

    resolved_paths = paths if paths is not None else _default_paths()
    disk = check_disk_quotas(
        {name: Path(p) for name, p in resolved_paths.items()}, quota_gb
    )
    status = "degraded" if disk["over_quota"] else "ok"
    return {
        "timestamp": _utc_now_iso(),
        "modules": modules,
        "disk": disk,
        "status": status,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(health_summary(), indent=2))
