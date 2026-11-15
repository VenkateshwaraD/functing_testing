"""LangGraph adapter: run a compiled graph, convert events to a Trajectory.

Duck-typed over ``astream_events`` payloads — no hard dependency on
langchain/langgraph imports, so a missing extra degrades to ConfigError,
not an import crash. Handles handoffs (multi-agent graphs) by tracking
node switches.
"""

from __future__ import annotations

from typing import Any

from aitestkit.exceptions import ConfigError
from aitestkit.models import Trajectory
from aitestkit.tracing.emitter import TraceEmitter


def langgraph_trajectory(events: list[dict[str, Any]]) -> Trajectory:
    """Convert astream_events dicts into a schema-versioned Trajectory."""
    emitter = TraceEmitter()
    for event in events:
        kind = event.get("event", "")
        name = event.get("name")
        data = event.get("data") or {}
        _convert(kind, name, data, emitter)
    if not emitter.trajectory.finish_reason:
        emitter.finish("done")
    return emitter.trajectory


def _convert(kind: str, name: str | None, data: dict[str, Any], emitter: TraceEmitter) -> None:
    if kind == "on_tool_start":
        emitter.tool_call(name or "tool", args=data.get("input") or {})
    elif kind == "on_tool_end":
        emitter.emit(
            "observation", name=name, result=_safe(data.get("output"))
        )
    elif kind in ("on_chat_model_stream", "on_llm_stream"):
        pass  # token spam — llm_call start/end carry the useful bits
    elif kind in ("on_chat_model_start", "on_llm_start"):
        emitter.llm_call(str(data.get("name") or name or "model"))
    elif kind in ("on_chain_start", "on_chain_end") and name:
        if name not in {"LangGraph", "__start__", "__end__"}:
            emitter.emit("handoff" if kind == "on_chain_start" else "message",
                         to_agent=name if kind == "on_chain_start" else None,
                         content=str(data.get("output"))[:200] if kind == "on_chain_end" else None,
                         agent_id=name)
    elif kind == "on_chain_error":
        emitter.emit("error", name=name, error=_safe(data))


async def invoke_langgraph(
    graph: Any, case_input: Any, *, config: dict[str, Any] | None = None
) -> tuple[Any, Trajectory]:
    """Run a compiled LangGraph app, returning (final_state, trajectory).

    ``graph`` is a compiled graph object with ``astream_events`` (v2 API).
    """
    if not hasattr(graph, "astream_events"):
        raise ConfigError(
            "langgraph target expects a compiled graph with astream_events; "
            "install aitestkit[langgraph]"
        )
    events: list[dict[str, Any]] = []
    final: Any = None
    kwargs: dict[str, Any] = {"version": "v2"}
    if config:
        kwargs["config"] = config
    try:
        async for event in graph.astream_events(case_input, **kwargs):
            events.append(event)
            if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                final = event.get("data", {}).get("output")
    except TypeError as e:
        raise ConfigError(f"langgraph stream failed — wrong graph object? {e}") from e
    return final, langgraph_trajectory(events)


def _safe(value: Any) -> Any:
    """Best-effort JSON-safe conversion of langchain objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return getattr(value, "content", str(value))
