"""Central configuration for NOVA AI Infrastructure.

Loads ``configs/config.yaml``, applies ``NOVA_`` environment-variable overrides,
and validates the result with clear error messages.

Env override format: ``NOVA_<SECTION>_<KEY>`` (case-insensitive), where the
section is matched against top-level config keys and the remainder is the
(possibly nested) key path with single underscores preserved:

    NOVA_MODULES_TEACHER=false            -> modules.teacher = False
    NOVA_SANDBOX_MAX_DOWNLOAD_BYTES=4096  -> sandbox.max_download_bytes = 4096
    NOVA_SANDBOX_ALLOWLIST="a.org,b.org"  -> sandbox.allowlist = ["a.org", "b.org"]

Types are coerced from the existing config value's type; unknown keys are
parsed smartly (true/false/int/float fallbacks, otherwise string).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "NOVA_"

#: Modules known to the infrastructure. Each maps to configs/config.yaml `modules.<name>`.
KNOWN_MODULES = (
    "sandbox",
    "crawlers",
    "ingestion",
    "deduplication",
    "curation",
    "teacher",
    "training",
    "evaluation",
    "monitoring",
    "security",
)


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or invalid."""


class NovaConfig(dict):
    """A dict subclass wrapping the merged configuration."""

    def is_enabled(self, name: str) -> bool:
        """Whether a module is enabled via ``modules.<name>``."""
        return bool(self.get("modules", {}).get(name, False))

    def section(self, name: str) -> dict:
        """Return a section (e.g. ``config.section("sandbox")``) as a plain dict."""
        value = self.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"config section {name!r} must be a mapping")
        return dict(value)


def default_config_path() -> Path:
    """Repo-root-relative default: <repo>/configs/config.yaml."""
    return Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"


def _smart_parse(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "yes", "1"):
        return True
    if lowered in ("false", "no", "0"):
        return False
    for caster in (int, float):
        try:
            return caster(value.strip())
        except ValueError:
            pass
    return value


def _coerce(raw: str, current: Any) -> Any:
    """Coerce an env string to the type of the existing config value."""
    text = raw.strip()
    if isinstance(current, bool):
        return _smart_parse(text) if isinstance(_smart_parse(text), bool) else current
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(text)
        except ValueError:
            raise ConfigError(f"cannot coerce {raw!r} to int")
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError:
            raise ConfigError(f"cannot coerce {raw!r} to float")
    if isinstance(current, list):
        return [item.strip() for item in text.split(",") if item.strip()]
    return _smart_parse(text)


def _set_nested(node: dict, key_path: str, raw: str) -> None:
    """Set ``key_path`` inside ``node``, tolerating underscores inside key names.

    Tries the full path as a direct key first, then longest-prefix nested matches.
    Unknown keys are stored smart-parsed.
    """
    if key_path in node:
        node[key_path] = _coerce(raw, node[key_path])
        return
    parts = key_path.split("_")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:i])
        rest_key = "_".join(parts[i:])
        if prefix in node and isinstance(node[prefix], dict):
            _set_nested(node[prefix], rest_key, raw)
            return
    node[key_path] = _smart_parse(raw)


def _apply_env_overrides(config: dict, env: dict[str, str] | None = None) -> dict:
    """Apply NOVA_ env vars onto the config dict (mutates and returns it)."""
    env = os.environ if env is None else env
    sections = [k for k in config.keys() if isinstance(config[k], dict)]
    for var, raw in env.items():
        if not var.startswith(ENV_PREFIX):
            continue
        rest = var[len(ENV_PREFIX):].lower()
        # Longest section-name prefix match so keys may contain underscores.
        section = next(
            (s for s in sorted(sections, key=len, reverse=True)
             if rest == s or rest.startswith(s + "_")),
            None,
        )
        if section is None:
            continue  # unknown section: ignore rather than crash
        key_path = rest[len(section):].lstrip("_")
        if not key_path:
            continue
        _set_nested(config[section], key_path, raw)
    return config


def _validate(config: NovaConfig) -> None:
    errors: list[str] = []
    modules = config.get("modules", {})
    if not isinstance(modules, dict):
        errors.append("'modules' must be a mapping of module name to boolean")
    else:
        for name in KNOWN_MODULES:
            if name in modules and not isinstance(modules[name], bool):
                errors.append(f"'modules.{name}' must be a boolean")
        for name in modules:
            if name not in KNOWN_MODULES:
                errors.append(f"unknown module {name!r} in 'modules'")

    sandbox = config.get("sandbox", {})
    if isinstance(sandbox, dict):
        for key in ("max_download_bytes", "timeout_seconds", "max_retries",
                    "requests_per_minute"):
            if key in sandbox and not isinstance(sandbox[key], (int, float)):
                errors.append(f"'sandbox.{key}' must be a number")
        if sandbox.get("max_download_bytes", 1) <= 0:
            errors.append("'sandbox.max_download_bytes' must be positive")
        if sandbox.get("timeout_seconds", 1) <= 0:
            errors.append("'sandbox.timeout_seconds' must be positive")

    training = config.get("training", {})
    if isinstance(training, dict):
        for key in ("batch_size", "gradient_accumulation_steps", "sequence_length"):
            if key in training and (
                not isinstance(training[key], int) or training[key] <= 0
            ):
                errors.append(f"'training.{key}' must be a positive integer")
        if "learning_rate" in training and not isinstance(
            training["learning_rate"], (int, float)
        ):
            errors.append("'training.learning_rate' must be a number")

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def load_config(path: str | Path | None = None,
                env: dict[str, str] | None = None) -> NovaConfig:
    """Load, override, and validate the central configuration."""
    cfg_path = Path(path) if path else default_config_path()
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {cfg_path} is not valid YAML: {exc}")
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {cfg_path} must contain a YAML mapping")
    merged = _apply_env_overrides(raw, env)
    config = NovaConfig(merged)
    _validate(config)
    return config
