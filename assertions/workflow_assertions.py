"""Workflow assertions — multi-agent handoffs, budgets, error-free runs."""

from __future__ import annotations

from typing import Any

from assertions._trajectory import iter_events


def assert_handoffs(trajectory: Any, expected_agents: list[str]) -> None:
    """Assert handoffs reached the expected agents, in order."""
    actual = [
        str(p.get("to_agent"))
        for t, p, _ in iter_events(trajectory)
        if t == "handoff"
    ]
    if actual != expected_agents:
        raise AssertionError(f"handoffs {actual} != expected {expected_agents}")


def assert_agent_participated(trajectory: Any, agent_id: str) -> None:
    """Assert at least one event was emitted by the named agent."""
    if not any(agent == agent_id for _, _, agent in iter_events(trajectory)):
        raise AssertionError(f"agent {agent_id!r} emitted no events")


def assert_step_budget(trajectory: Any, *, max_steps: int) -> None:
    """Assert the trajectory stayed within a step budget."""
    count = len(iter_events(trajectory))
    if count > max_steps:
        raise AssertionError(f"{count} events exceeded budget {max_steps}")


def assert_no_tool_errors(trajectory: Any) -> None:
    """Assert no tool_call event carries an error."""
    failures = [
        str(p.get("name"))
        for t, p, _ in iter_events(trajectory)
        if t == "tool_call" and p.get("error")
    ]
    if failures:
        raise AssertionError(f"tool errors on: {failures}")
