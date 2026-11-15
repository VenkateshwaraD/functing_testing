"""Agent assertions — tool calls, trajectory ordering, safety gates."""

from __future__ import annotations

from typing import Any

from assertions._trajectory import finish_reason, iter_events


def _calls(trajectory: Any) -> list[dict]:
    return [p for t, p, _ in iter_events(trajectory) if t == "tool_call"]


def assert_tool_called(
    trajectory: Any, name: str, *, args: dict | None = None
) -> None:
    """Assert a tool ran; if args given, they must be a subset of actual."""
    for call in _calls(trajectory):
        if call.get("name") != name:
            continue
        actual = dict(call.get("args") or {})
        if args is None or all(actual.get(k) == v for k, v in args.items()):
            return
        raise AssertionError(
            f"{name} args mismatch: expected {args} ⊆ {actual}"
        )
    raise AssertionError(f"tool {name!r} was never called")


def assert_tool_not_called(trajectory: Any, name: str) -> None:
    """Assert a forbidden tool was never invoked."""
    if any(c.get("name") == name for c in _calls(trajectory)):
        raise AssertionError(f"forbidden tool {name!r} was invoked")


def assert_tool_order(trajectory: Any, names: list[str]) -> None:
    """Assert the tools ran in the given order (as a subsequence)."""
    called = [c.get("name") for c in _calls(trajectory)]
    position = 0
    for expected in names:
        while position < len(called) and called[position] != expected:
            position += 1
        if position == len(called):
            raise AssertionError(f"{expected!r} missing/late in {called}")
        position += 1


def assert_confirmation_precedes(trajectory: Any, tool_name: str) -> None:
    """Assert a confirmation event precedes a destructive tool call."""
    confirmed = False
    for event_type, payload, _ in iter_events(trajectory):
        if event_type == "confirmation" or (
            event_type == "message"
            and payload.get("role") == "confirmation"
        ):
            confirmed = True
        if event_type == "tool_call" and payload.get("name") == tool_name:
            if not confirmed:
                raise AssertionError(
                    f"{tool_name!r} ran without prior confirmation"
                )
            return
    raise AssertionError(f"{tool_name!r} was never called")


def assert_terminated(trajectory: Any) -> None:
    """Assert the run finished cleanly (done, not timeout/max_steps)."""
    reason = finish_reason(trajectory)
    if reason not in (None, "done"):
        raise AssertionError(f"run did not terminate cleanly: {reason}")


def assert_max_tool_repeats(trajectory: Any, *, max_repeats: int = 3) -> None:
    """Assert no single tool ran more than max_repeats (loop guard)."""
    counts: dict[str, int] = {}
    for call in _calls(trajectory):
        name = str(call.get("name"))
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > max_repeats:
            raise AssertionError(
                f"loop detected: {name!r} called {counts[name]} times"
            )
