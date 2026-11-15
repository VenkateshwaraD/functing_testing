"""Target adapter protocol: how aitestkit invokes the system under test."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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
