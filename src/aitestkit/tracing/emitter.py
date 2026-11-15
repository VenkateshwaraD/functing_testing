"""TraceEmitter: the framework-agnostic way a SUT reports its trajectory.

Passed to targets as ``trace_sink`` — a callable the SUT (or its adapter)
invokes per step. Works for any agent framework: the SUT emits semantic
events (thought, tool_call, handoff, memory_op, retrieval...) and the
emitter assembles a schema-versioned Trajectory for metrics and reports.
"""

from __future__ import annotations

import time
from typing import Any

from aitestkit.models import TraceEvent, Trajectory


class TraceEmitter:
    """Collects TraceEvents; produces the run Trajectory."""

    def __init__(self, *, agent_id: str | None = None) -> None:
        self._events: list[TraceEvent] = []
        self._finish_reason: str | None = None
        self.default_agent = agent_id

    def emit(
        self,
        event_type: str,
        *,
        agent_id: str | None = None,
        **payload: Any,
    ) -> TraceEvent:
        """Record one event; returns it for convenience."""
        event = TraceEvent(
            type=event_type,
            agent_id=agent_id or self.default_agent,
            step_index=len(self._events),
            timestamp=time.time(),
            payload=payload,
        )
        self._events.append(event)
        return event

    def __call__(self, event_type: str, **payload: Any) -> TraceEvent:
        """Sink protocol: ``trace_sink('tool_call', name=..., args=...)``."""
        return self.emit(event_type, **payload)

    def thought(self, text: str, *, agent_id: str | None = None) -> TraceEvent:
        return self.emit("thought", agent_id=agent_id, text=text)

    def tool_call(
        self,
        name: str,
        *,
        args: dict[str, Any] | None = None,
        result: Any | None = None,
        error: str | None = None,
        agent_id: str | None = None,
    ) -> TraceEvent:
        return self.emit(
            "tool_call", agent_id=agent_id, name=name,
            args=args or {}, result=result, error=error,
        )

    def handoff(self, to_agent: str, *, from_agent: str | None = None) -> TraceEvent:
        return self.emit("handoff", agent_id=from_agent, to_agent=to_agent)

    def memory_op(self, op: str, *, key: str, value: Any = None) -> TraceEvent:
        return self.emit("memory_op", op=op, key=key, value=value)

    def retrieval(self, chunks: list[str], *, agent_id: str | None = None) -> TraceEvent:
        return self.emit("retrieval", agent_id=agent_id, chunks=chunks)

    def llm_call(self, model: str, *, agent_id: str | None = None, **kw: Any) -> TraceEvent:
        return self.emit("llm_call", agent_id=agent_id, model=model, **kw)

    def message(self, role: str, content: str, *, agent_id: str | None = None) -> TraceEvent:
        return self.emit("message", agent_id=agent_id, role=role, content=content)

    def finish(self, reason: str = "done") -> None:
        """Mark run termination: done|max_steps|timeout|error|cancelled."""
        self._finish_reason = reason

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(events=list(self._events), finish_reason=self._finish_reason)

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        """A present emitter is truthy even with zero events — keeps the
        ``trace_sink or _NullSink()`` / ``if trace_sink:`` idiom working."""
        return True
