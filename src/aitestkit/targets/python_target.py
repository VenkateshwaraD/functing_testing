"""Python target: invoke an in-repo callable as the system under test.

``python: package.module:function``. Sync functions run in an executor;
async functions are awaited. The callable may return a string, a
``TargetOutput``, or a dict shaped like one — optionally emitting trace
events through a ``TraceEmitter`` sink passed as ``trace_sink``.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from typing import Any

from aitestkit.exceptions import ConfigError, TargetError
from aitestkit.models import TargetOutput, TestCase, Trajectory, Usage


def load_callable(spec: str) -> Any:
    """Import ``module:function`` and return the callable."""
    if ":" not in spec:
        raise ConfigError(
            f"python target {spec!r} must be 'module:function' (e.g. myapp.chat:run)"
        )
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ConfigError(f"cannot import {module_name!r}: {e}") from e
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ConfigError(f"{spec!r} does not resolve to a callable")
    return func


def _coerce_output(result: Any) -> TargetOutput:
    """Normalize arbitrary callable return values into TargetOutput."""
    if isinstance(result, TargetOutput):
        return result
    if isinstance(result, str):
        return TargetOutput(output=result)
    if isinstance(result, dict):
        trajectory = result.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory = Trajectory.model_validate(trajectory)
        usage_raw = result.get("usage") or result.get("sut_usage") or {}
        usage = usage_raw if isinstance(usage_raw, Usage) else Usage.model_validate(usage_raw)
        return TargetOutput(
            output=result.get("output") or result.get("text"),
            trajectory=trajectory,
            retrieval_context=result.get("retrieval_context") or result.get("context"),
            usage=usage,
        )
    return TargetOutput(output=str(result))


class PythonTarget:
    """Target adapter backed by a Python callable in the test project."""

    name = "python"

    def __init__(self, *, spec: str, kwargs: dict[str, Any] | None = None) -> None:
        self.spec = spec
        self._func = load_callable(spec)
        self._extra_kwargs = kwargs or {}

    async def invoke(
        self,
        case: TestCase,
        *,
        model: str | None = None,
        provider: str | None = None,
        tool_mocks: dict[str, Any] | None = None,
        session_id: str | None = None,
        trace_sink: Any | None = None,
    ) -> TargetOutput:
        kwargs = self._build_kwargs(
            case,
            model=model,
            provider=provider,
            tool_mocks=tool_mocks,
            session_id=session_id,
            trace_sink=trace_sink,
        )
        start = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(**kwargs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._func(**kwargs))
        except Exception as e:  # noqa: BLE001 — surfaced as TargetError below
            raise TargetError(f"python target {self.spec} raised: {e}") from e
        output = _coerce_output(result)
        output.latency_ms = (time.perf_counter() - start) * 1000
        return output

    def _build_kwargs(
        self,
        case: TestCase,
        *,
        model: str | None,
        provider: str | None,
        tool_mocks: dict[str, Any] | None,
        session_id: str | None,
        trace_sink: Any | None,
    ) -> dict[str, Any]:
        """Pass only kwargs the callable actually accepts (plus input always)."""
        signature = inspect.signature(self._func)
        candidates: dict[str, Any] = {
            "input": case.input,
            "context": case.context,
            "model": model,
            "provider": provider,
            "tool_mocks": tool_mocks,
            "session_id": session_id,
            "trace_sink": trace_sink,
            **self._extra_kwargs,
        }
        kwargs = {"input": case.input}
        for key, value in candidates.items():
            if key == "input":
                continue
            if key in signature.parameters and value is not None:
                kwargs[key] = value
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            kwargs.update({k: v for k, v in candidates.items() if v is not None})
        return kwargs
