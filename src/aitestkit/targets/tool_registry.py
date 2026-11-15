"""ToolRegistry: tools the SUT can call, with fault injection.

Used by agent targets to expose tools during a run — every call is recorded
as a ``ToolCall`` for the trajectory, and per-tool faults (error, timeout,
malformed results) let suites test recovery and loop behavior without
touching the agent code. ``requires_confirmation`` models tools a human
must approve before destructive actions.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from aitestkit.exceptions import ConfigError, TargetError
from aitestkit.models import ToolCall


class ToolFault:
    """Fault injection spec for one tool."""

    def __init__(
        self,
        *,
        error: str | None = None,
        timeout_ms: float | None = None,
        malformed: bool = False,
        times: int | None = None,
    ) -> None:
        self.error = error
        self.timeout_ms = timeout_ms
        self.malformed = malformed
        self._remaining = times  # None = always fault; N = first N calls

    def should_fire(self) -> bool:
        if self._remaining is None:
            return True
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True


class ToolRegistry:
    """Registry of callable tools + fault injection + call recording."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._confirm: set[str] = set()
        self._faults: dict[str, ToolFault] = {}
        self._mocks: dict[str, Any] = {}
        self.calls: list[ToolCall] = []

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        requires_confirmation: bool = False,
    ) -> None:
        """Register a tool callable by name."""
        self._tools[name] = fn
        if requires_confirmation:
            self._confirm.add(name)

    def inject_fault(self, name: str, fault: ToolFault) -> None:
        """Attach a fault to a tool (error/timeout/malformed)."""
        self._require_registered(name)
        self._faults[name] = fault

    def mock_result(self, name: str, value: Any) -> None:
        """Force a tool to return a canned value (tool mock)."""
        self._require_registered(name)
        self._mocks[name] = value

    def requires_confirmation(self, name: str) -> bool:
        """Whether the tool needs human confirmation before use."""
        return name in self._confirm

    def has(self, name: str) -> bool:
        """Whether a tool name is registered."""
        return name in self._tools

    async def invoke(
        self, name: str, args: dict[str, Any] | None = None, *, timeout_ms: float = 10_000
    ) -> Any:
        """Call a tool: inject faults, record the call, return the result."""
        self._require_registered(name)
        args = args or {}
        call = ToolCall(name=name, args=args, requires_confirmation=name in self._confirm)
        self.calls.append(call)
        try:
            call.result = await self._dispatch(name, args, timeout_ms=timeout_ms)
        except (TargetError, TimeoutError) as e:
            call.error = str(e)
            raise
        return call.result

    async def _dispatch(self, name: str, args: dict[str, Any], *, timeout_ms: float) -> Any:
        fault = self._faults.get(name)
        if fault and fault.should_fire():
            return await self._fire_fault(name, fault)
        if name in self._mocks:
            return self._mocks[name]
        fn = self._tools[name]
        if inspect.iscoroutinefunction(fn):
            return await asyncio.wait_for(fn(**args), timeout=timeout_ms / 1000)
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: fn(**args)), timeout=timeout_ms / 1000
        )

    async def _fire_fault(self, name: str, fault: ToolFault) -> Any:
        if fault.timeout_ms is not None:
            await asyncio.sleep(fault.timeout_ms / 1000)
            raise TimeoutError(f"tool {name} injected timeout")
        if fault.error is not None:
            raise TargetError(f"tool {name} injected error: {fault.error}")
        if fault.malformed:
            return {"__malformed__": True, "tool": name}
        raise TargetError(f"tool {name} fault misconfigured")

    def _require_registered(self, name: str) -> None:
        if name not in self._tools:
            raise ConfigError(f"tool {name!r} not registered in ToolRegistry")


def build_registry(spec: dict[str, Any] | None) -> ToolRegistry | None:
    """Build a registry from a suite ``tools:`` config block.

    ``spec`` maps tool name -> {"function": "module:fn", "requires_confirmation": bool,
    "fault": {...}, "mock": value}.
    """
    if not spec:
        return None
    from aitestkit.targets.python_target import load_callable

    registry = ToolRegistry()
    for name, tool_cfg in spec.items():
        fn = load_callable(tool_cfg["function"])
        registry.register(
            name, fn, requires_confirmation=tool_cfg.get("requires_confirmation", False)
        )
        if "fault" in tool_cfg:
            registry.inject_fault(name, ToolFault(**tool_cfg["fault"]))
        if "mock" in tool_cfg:
            registry.mock_result(name, tool_cfg["mock"])
    return registry
