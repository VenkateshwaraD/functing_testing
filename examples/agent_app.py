"""Demo agent — emits a trajectory via trace_sink for agentic metrics.

Stands in for a real ReAct agent: decides on a tool from keywords, records
the thought/action/observation, answers from the result. Deterministic so
the example suite replays offline.
"""

from __future__ import annotations

from typing import Any


def _tool_search(query: str) -> str:
    results = {
        "weather": "sunny, 22C",
        "hotel": "Grand Hotel, $120/night",
        "flight": "AF123, departing 09:00",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return "no results"


def run(input: str, *, trace_sink: Any | None = None) -> dict:
    """Think -> act (tool) -> observe -> answer, emitting trace events."""
    class _Null:
        def __getattr__(self, name: str):  # noqa: ANN001
            return lambda *a, **k: None

    sink = trace_sink if trace_sink is not None else _Null()

    sink.thought(f"deciding what to do about: {input}")
    tool = "search" if any(
        word in input.lower() for word in ("weather", "hotel", "flight")
    ) else "none"
    sink.emit("action", text=f"use tool: {tool}")
    observation = _tool_search(input) if tool != "none" else ""
    if tool != "none":
        sink.tool_call("search", args={"query": input})
        sink.emit("observation", text=observation)
    answer = f"here's what I found: {observation}" if observation else (
        f"I can't help with {input!r} directly"
    )
    sink.message("assistant", answer)
    return {
        "output": answer,
        "tool_calls": (
            [{"name": "search", "arguments": {"query": input}}]
            if tool != "none"
            else []
        ),
    }
