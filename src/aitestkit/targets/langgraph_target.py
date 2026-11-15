"""LangGraph target: invoke a compiled graph as the system under test.

``langgraph: package.module:graph`` resolves to a compiled graph object
(with ``astream_events``); runs it, converts the event stream into a
schema-versioned Trajectory, and extracts the final answer. Optional
extra: ``aitestkit[langgraph]``.
"""

from __future__ import annotations

import time
from typing import Any

from aitestkit.exceptions import ConfigError, TargetError
from aitestkit.models import TargetOutput, TestCase
from aitestkit.targets.python_target import load_callable


class LangGraphTarget:
    """Run a compiled LangGraph graph per case."""

    name = "langgraph"

    def __init__(self, *, spec: str, input_key: str = "input") -> None:
        self.spec = spec
        self.input_key = input_key
        self._graph = self._load(spec)

    @staticmethod
    def _load(spec: str) -> Any:
        graph = load_callable(spec)
        candidate = graph() if callable(graph) and not hasattr(graph, "astream_events") else graph
        if not hasattr(candidate, "astream_events"):
            raise ConfigError(
                f"{spec!r} must resolve to a compiled LangGraph graph "
                "(needs astream_events) or a zero-arg factory returning one"
            )
        return candidate

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
        from aitestkit.agentic.langgraph_adapter import invoke_langgraph

        payload: Any = case.input
        if isinstance(case.input, str):
            payload = {self.input_key: case.input}
        start = time.perf_counter()
        try:
            final, trajectory = await invoke_langgraph(self._graph, payload)
        except ConfigError:
            raise
        except Exception as e:  # noqa: BLE001 — surfaced as TargetError
            raise TargetError(f"langgraph target {self.spec} raised: {e}") from e
        output = TargetOutput(
            trajectory=trajectory,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        output.output = _final_answer(final)
        return output


def _final_answer(final: Any) -> str | None:
    """Pull the answer string out of a final graph state."""
    if final is None:
        return None
    if isinstance(final, str):
        return final
    if isinstance(final, dict):
        for key in ("output", "answer", "result", "messages"):
            if key in final:
                value = final[key]
                if isinstance(value, list) and value:
                    last = value[-1]
                    return getattr(last, "content", str(last))
                return str(value)
        return str(final)
    return getattr(final, "content", str(final))
