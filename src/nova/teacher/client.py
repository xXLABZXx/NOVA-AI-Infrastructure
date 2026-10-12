"""Teacher model client (e.g. DeepSeek) with hard spending limits.

The client talks to an OpenAI-compatible ``/chat/completions`` endpoint
using only the standard library (``urllib``). When no API key is
configured the client is *disabled* and every call returns a
``{"disabled": True, ...}`` dict instead of touching the network.

The API key is never included in returned dicts, logs, or exception
messages: all outbound text passes through redaction.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable


class TeacherDisabled(Exception):
    """Raised when the teacher cannot be used (no key, or budget exhausted)."""


def _redact(text: str, secret: str | None) -> str:
    """Replace any occurrence of ``secret`` in ``text`` with [REDACTED]."""
    if not secret or not text:
        return text
    return text.replace(secret, "[REDACTED]")


class TeacherClient:
    """Thin client for an OpenAI-compatible teacher-model API."""

    def __init__(
        self,
        *,
        provider: str = "deepseek",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        api_key: str | None = None,
        max_calls_per_run: int = 50,
        max_tokens_per_call: int = 2000,
        temperature: float = 0.2,
        timeout_seconds: int = 60,
        auth_hook: Callable[[urllib.request.Request], None] | None = None,
    ) -> None:
        """Create a teacher client.

        ``auth_hook`` is an optional callable applied to every outbound
        ``urllib`` request before it is sent. It exists for runtimes (like
        Hatch's Secure Vault) where the credential must be attached as a
        short-lived surrogate instead of a static ``Authorization`` header.
        When ``auth_hook`` is given, no raw API key is needed or stored.
        """
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.auth_hook = auth_hook
        self.max_calls_per_run = max_calls_per_run
        self.max_tokens_per_call = max_tokens_per_call
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.calls_made = 0

    # -- construction helpers -------------------------------------------
    @classmethod
    def from_env(cls, section: dict | None = None) -> "TeacherClient":
        """Build from environment (DEEPSEEK_API_KEY or TEACHER_API_KEY).

        Other parameters come from ``section`` (e.g. the ``teacher``
        config section).
        """
        section = dict(section or {})
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
            "TEACHER_API_KEY"
        )
        kwargs: dict = {"api_key": api_key}
        for key in (
            "provider",
            "base_url",
            "model",
            "max_calls_per_run",
            "max_tokens_per_call",
            "temperature",
            "timeout_seconds",
        ):
            if key in section:
                kwargs[key] = section[key]
        return cls(**kwargs)

    @classmethod
    def from_config(cls, section: dict) -> "TeacherClient":
        """Build from a config section dict (API key still comes from env)."""
        return cls.from_env(section)

    # -- state ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True when a credential is available (raw key or auth hook)."""
        return bool(self.api_key) or self.auth_hook is not None

    # -- HTTP plumbing ------------------------------------------------------
    def _post_json(
        self, url: str, payload: dict, headers: dict
    ) -> dict:
        """POST a JSON payload and parse the JSON response (urllib only).

        Separated from :meth:`complete` so tests can monkeypatch it.
        Never logs or raises with the API key included.
        """
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        if self.auth_hook is not None:
            self.auth_hook(request)
        try:
            with urllib.request.urlopen(request,
                                        timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise RuntimeError(
                _redact(f"teacher API HTTP {exc.code}: {detail}", self.api_key)
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                _redact(f"teacher API connection failed: {exc.reason}",
                        self.api_key)
            ) from exc
        return json.loads(raw)

    # -- main API -----------------------------------------------------------
    def complete(self, prompt: str, max_tokens: int | None = None) -> dict:
        """Send ``prompt`` to the teacher model and return a result dict.

        Returns ``{"disabled": True, "reason": ...}`` without any network
        access when no API key is configured. Raises :class:`TeacherDisabled`
        once ``max_calls_per_run`` calls have been made. The API key never
        appears in the returned dict or in raised exceptions.
        """
        if not self.enabled:
            return {"disabled": True, "reason": "no API key configured"}
        if self.calls_made >= self.max_calls_per_run:
            raise TeacherDisabled(
                f"spending limit reached: {self.calls_made} calls "
                f"(max {self.max_calls_per_run} per run)"
            )
        budget = self.max_tokens_per_call
        tokens = budget if max_tokens is None else min(max_tokens, budget)
        if tokens <= 0:
            raise ValueError("max_tokens must be positive")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": self.temperature,
        }
        headers = {}
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        # With an auth_hook the credential is attached per-request (e.g. a
        # short-lived surrogate); no static Authorization header is needed.

        self.calls_made += 1  # attempts count against the budget
        try:
            data = self._post_json(url, payload, headers)
        except TeacherDisabled:
            raise
        except Exception as exc:
            raise RuntimeError(
                _redact(f"teacher request failed: {exc}", self.api_key)
            ) from exc

        content = ""
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            content = ""
        # Build the result only from parsed fields: the key never enters it.
        return {
            "content": content,
            "model": data.get("model", self.model) if isinstance(data, dict) else self.model,
            "usage": data.get("usage", {}) if isinstance(data, dict) else {},
            "calls_made": self.calls_made,
        }
