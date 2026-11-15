"""Internal helpers — normalize trajectories to (type, payload, agent_id)."""

from __future__ import annotations

from typing import Any


def iter_events(trajectory: Any) -> list[tuple[str, dict, str | None]]:
    """Return (type, payload, agent_id) for Trajectory, dict, or event list."""
    if trajectory is None:
        return []
    events = getattr(trajectory, "events", None)
    if events is None and isinstance(trajectory, dict):
        events = trajectory.get("events", [])
    if events is None:
        events = trajectory if isinstance(trajectory, list) else []
    return [(_type(e), _payload(e), _agent(e)) for e in events]


def finish_reason(trajectory: Any) -> str | None:
    """Extract finish_reason from a Trajectory or serialized dict."""
    if isinstance(trajectory, dict):
        return trajectory.get("finish_reason")
    return getattr(trajectory, "finish_reason", None)


def _type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type", ""))
    return str(getattr(event, "type", ""))


def _payload(event: Any) -> dict:
    if isinstance(event, dict):
        return dict(event.get("payload", {}))
    return dict(getattr(event, "payload", {}) or {})


def _agent(event: Any) -> str | None:
    agent = (
        event.get("agent_id")
        if isinstance(event, dict)
        else getattr(event, "agent_id", None)
    )
    return str(agent) if agent else None
