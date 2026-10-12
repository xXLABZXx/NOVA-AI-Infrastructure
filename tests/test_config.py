"""Tests for nova.config: loading, env overrides, coercion, validation."""

from pathlib import Path

import pytest

from nova.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def test_loads_default_config_ok():
    config = load_config(DEFAULT_CONFIG, env={})
    assert config.is_enabled("security") is True
    assert config.is_enabled("monitoring") is True
    assert config.is_enabled("teacher") is False
    assert config.section("sandbox")["max_download_bytes"] == 5242880


def test_env_override_flips_module_flag(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("modules:\n  teacher: true\n", encoding="utf-8")
    config = load_config(cfg_file, env={"NOVA_MODULES_TEACHER": "false"})
    assert config.is_enabled("teacher") is False


def test_env_override_coerces_int(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "sandbox:\n  max_download_bytes: 5242880\n", encoding="utf-8"
    )
    config = load_config(cfg_file, env={"NOVA_SANDBOX_MAX_DOWNLOAD_BYTES": "4096"})
    value = config.section("sandbox")["max_download_bytes"]
    assert value == 4096
    assert isinstance(value, int)


def test_invalid_module_value_raises_config_error(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text('modules:\n  teacher: "yes-please"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(cfg_file, env={})
    assert "modules.teacher" in str(excinfo.value)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.yaml", env={})
