"""Dataset version manifests: content hashes for reproducibility.

``write_manifest`` records the sha256 of every ``*.jsonl`` file in a
directory so downstream consumers can verify they are training on exactly
the bytes that were approved. ``verify_manifest`` re-hashes and reports
any drift.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    directory: str | Path,
    version: str,
    extra: dict | None = None,
) -> Path:
    """Hash every ``*.jsonl`` in ``directory`` and write ``manifest.json``.

    The manifest contains ``version``, ``created_at``, and a ``files`` list
    of ``{name, sha256, bytes, records}`` dicts, plus any ``extra`` keys.
    Returns the path of the written manifest.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    for jsonl_path in sorted(directory.glob("*.jsonl")):
        records = 0
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records += 1
        files.append(
            {
                "name": jsonl_path.name,
                "sha256": _sha256_file(jsonl_path),
                "bytes": jsonl_path.stat().st_size,
                "records": records,
            }
        )
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    if extra:
        manifest.update(dict(extra))
    out = directory / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def verify_manifest(manifest_path: str | Path) -> tuple[bool, list[str]]:
    """Re-hash every file listed in ``manifest_path``.

    Returns ``(ok, problems)`` where ``ok`` is True only when every file
    exists and matches its recorded hash.
    """
    manifest_path = Path(manifest_path)
    problems: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"cannot read manifest {manifest_path}: {exc}"]
    base = manifest_path.parent
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return False, ["manifest 'files' is not a list"]
    for entry in files:
        name = entry.get("name")
        target = base / name if name else None
        if target is None or not target.is_file():
            problems.append(f"missing file: {name}")
            continue
        actual = _sha256_file(target)
        if actual != entry.get("sha256"):
            problems.append(
                f"hash mismatch: {name} "
                f"(expected {entry.get('sha256')}, got {actual})"
            )
    return (len(problems) == 0), problems
