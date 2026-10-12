"""Tests for the nova.monitoring package (hermetic: no network, no APIs)."""

import json

from nova.monitoring import (
    EventLogger,
    check_disk_quotas,
    get_logger,
    health_summary,
)


# --- logger -----------------------------------------------------------------


def test_get_logger_returns_event_logger(tmp_path):
    logger = get_logger(tmp_path / "logs")
    assert isinstance(logger, EventLogger)
    assert logger.path == tmp_path / "logs" / "nova.jsonl"


def test_event_logger_redacts_sensitive_fields(tmp_path):
    logger = get_logger(tmp_path / "logs")
    record = logger.log("deploy", api_key="sk-secret-123", user="cleo")
    assert record["api_key"] == "[REDACTED]"
    assert record["user"] == "cleo"
    parsed = json.loads(logger.path.read_text(encoding="utf-8").strip())
    assert parsed["api_key"] == "[REDACTED]"


def test_event_logger_redacts_secret_like_string_values(tmp_path):
    logger = get_logger(tmp_path / "logs")
    record = logger.log(
        "teacher_call", note="DEEPSEEK_API_KEY=sk-abcdef1234567890"
    )
    assert record["note"] == "[REDACTED]"
    # Case-insensitive key matching: "Password" and nested structures too.
    record2 = logger.log(
        "login", Password="hunter2", nested={"auth": "abc"}, items=["ok"]
    )
    assert record2["Password"] == "[REDACTED]"
    assert record2["nested"] == {"auth": "[REDACTED]"}
    assert record2["items"] == ["ok"]


def test_event_logger_writes_valid_jsonl_with_timestamp(tmp_path):
    logger = get_logger(tmp_path / "logs", name="events")
    logger.log("first", n=1)
    logger.log("second", n=2)
    lines = logger.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event"] == "first" and first["n"] == 1
    assert second["event"] == "second" and second["n"] == 2
    assert "timestamp" in first and "timestamp" in second


# --- quotas -----------------------------------------------------------------


def test_check_disk_quotas_reports_usage(tmp_path):
    data_file = tmp_path / "data.bin"
    data_file.write_bytes(b"x" * 2048)
    result = check_disk_quotas({"work": tmp_path}, quota_gb=1.0)
    assert result["quota_gb"] == 1.0
    assert result["paths"]["work"]["bytes"] >= 2048
    assert result["paths"]["work"]["gb"] >= 0.0
    assert result["total_gb"] >= result["paths"]["work"]["gb"]
    assert result["over_quota"] is False


def test_check_disk_quotas_over_quota_with_zero_quota(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"x" * 100)
    result = check_disk_quotas({"work": tmp_path}, quota_gb=0)
    assert result["paths"]["work"]["bytes"] > 0
    assert result["over_quota"] is True


def test_check_disk_quotas_missing_path_counts_zero():
    result = check_disk_quotas({"gone": "/no/such/dir/anywhere"}, quota_gb=1.0)
    assert result["paths"]["gone"]["bytes"] == 0
    assert result["over_quota"] is False


# --- health -----------------------------------------------------------------


def test_health_summary_returns_expected_keys(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"x" * 100)
    config = {
        "modules": {"security": True, "monitoring": True, "teacher": False},
        "monitoring": {"disk_quota_gb": 50},
    }
    summary = health_summary(config=config, paths={"work": tmp_path})
    assert set(summary) == {"timestamp", "modules", "disk", "status"}
    assert summary["modules"]["security"] is True
    assert summary["modules"]["teacher"] is False
    assert summary["disk"]["over_quota"] is False
    assert summary["status"] == "ok"


def test_health_summary_degraded_when_over_quota(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"x" * 100)
    config = {"modules": {}, "monitoring": {"disk_quota_gb": 0}}
    summary = health_summary(config=config, paths={"work": tmp_path})
    assert summary["status"] == "degraded"
    assert summary["disk"]["over_quota"] is True
