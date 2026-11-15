"""Target adapter protocol: how aitestkit invokes the system under test."""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, runtime_checkable

from aitestkit.exceptions import ConfigError
from aitestkit.models import TargetOutput, TestCase


@runtime_checkable
class TargetAdapter(Protocol):
    """Invoke the system under test for one case."""

    name: str

    async def invoke(
        self,
        case: TestCase,
        *,
        model: str | None = None,
        provider: str | None = None,
        tool_mocks: dict[str, Any] | None = None,
        session_id: str | None = None,
        trace_sink: Any | None = None,
        tool_registry: Any | None = None,
        prompt: str | None = None,
        prompt_name: str | None = None,
    ) -> TargetOutput:
        """Run the SUT for ``case`` and return its output + telemetry."""
        ...


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str) -> str:
    """Expand ${VAR} references — keeps secrets out of suite files.

    Unset vars raise :class:`ConfigError`: silently forwarding a literal
    ``${VAR}`` as a header/URL value fails at best and misleads at worst.
    """

    def _sub(match: re.Match[str]) -> str:
        var = match.group(1)
        if var not in os.environ:
            raise ConfigError(
                f"env var ${{{var}}} referenced but not set — "
                "export it before running"
            )
        return os.environ[var]

    return _ENV_RE.sub(_sub, value)


# Env vars safe to hand a spawned SUT — never provider keys or arbitrary
# parent state. Suite authors pass extras explicitly via ``env:`` kwargs.
_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TERM",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SYSTEMROOT",
        "COMSPEC",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV_PROMPT",
    }
)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a minimal env for a spawned subprocess (command/mcp targets).

    Inheriting ``os.environ`` wholesale leaks provider keys (``OPENAI_API_KEY``
    etc.) into the child. Only allowlisted vars, ``LC_*``, ``AITESTKIT_*``
    config vars, and explicit ``env:`` kwargs reach the process; ``env:``
    values expand ``${VAR}`` so secrets still stay out of suite files.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _CHILD_ENV_ALLOWLIST
        or k.startswith("LC_")
        or k.startswith("AITESTKIT_")
    }
    for key, value in (extra or {}).items():
        env[key] = expand_env(value)
    return env
