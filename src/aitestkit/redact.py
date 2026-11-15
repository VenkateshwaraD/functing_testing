"""Shared redaction boundary for every persisted aitestkit output.

Applied to logs, reports, JSONL streams, cassettes, and baselines — not just
logs. Built-in secret patterns plus configured environment values plus
optional suite regexes are replaced with ``[REDACTED]``.
"""

from __future__ import annotations

import os
import re
from typing import Any

_BUILTIN_PATTERNS: list[str] = [
    r"sk-[A-Za-z0-9_-]{20,}",
    r"sk-ant-[A-Za-z0-9_-]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"gho_[A-Za-z0-9]{20,}",
    r"AIza[0-9A-Za-z_-]{20,}",
    r"Bearer [A-Za-z0-9._\-]{10,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----",
]

# Env vars whose values must never appear in persisted output.
_SECRET_ENV_NAMES = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_API_KEY",
    "PINECONE_API_KEY",
    "COHERE_API_KEY",
    "HF_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
]

_REDACTED = "[REDACTED]"


class Redactor:
    """Redact secrets from text and structured data.

    Args:
        extra_patterns: additional regexes (from suite config).
        env_names: extra env vars whose values should be redacted.
        enabled: when False, redact() is a no-op (`--no-redact` escape).
    """

    def __init__(
        self,
        *,
        extra_patterns: list[str] | None = None,
        env_names: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        patterns = [*_BUILTIN_PATTERNS, *(extra_patterns or [])]
        self._regexes = [re.compile(p) for p in patterns]
        names = [*_SECRET_ENV_NAMES, *(env_names or [])]
        self._env_values = {
            v for n in names if (v := os.environ.get(n)) and len(v) >= 8
        }
        self.enabled = enabled

    def redact_text(self, text: str) -> str:
        """Return text with secrets replaced by ``[REDACTED]``."""
        if not self.enabled or not text:
            return text
        for value in self._env_values:
            text = text.replace(value, _REDACTED)
        for regex in self._regexes:
            text = regex.sub(_REDACTED, text)
        return text

    def redact(self, data: Any) -> Any:
        """Deep-redact strings inside dicts/lists; other types pass through."""
        if not self.enabled:
            return data
        if isinstance(data, str):
            return self.redact_text(data)
        if isinstance(data, dict):
            return {k: self.redact(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self.redact(v) for v in data]
        return data
