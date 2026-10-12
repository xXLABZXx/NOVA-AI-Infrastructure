"""Security primitives: validation, safe JSON, secret scanning, injection guard, audit."""

from nova.security.audit import write_audit_event
from nova.security.injection_guard import GuardResult, guard_text
from nova.security.safe_json import safe_dumps, safe_load_file, safe_loads
from nova.security.secret_scan import (
    SECRET_PATTERNS,
    SecretFinding,
    scan_file_for_secrets,
    scan_text_for_secrets,
)
from nova.security.validators import (
    ValidationError,
    clamp_int,
    validate_domain,
    validate_record,
    validate_url,
)

__all__ = [
    "ValidationError",
    "validate_url",
    "validate_domain",
    "validate_record",
    "clamp_int",
    "safe_loads",
    "safe_load_file",
    "safe_dumps",
    "scan_text_for_secrets",
    "scan_file_for_secrets",
    "SecretFinding",
    "SECRET_PATTERNS",
    "guard_text",
    "GuardResult",
    "write_audit_event",
]
