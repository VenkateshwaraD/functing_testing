"""HTTP target: call a REST endpoint as the system under test.

``request_template`` interpolates ``{input}``, ``{context}``, ``{model}``,
``{session_id}`` into JSON fields; ``response_path`` / ``context_path`` /
``tools_path`` extract structured fields with a dotted-path resolver.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from aitestkit.exceptions import TargetError
from aitestkit.models import TargetOutput, TestCase


class HttpTarget:
    """POST the case to a JSON endpoint and extract fields from the reply."""

    name = "http"

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        request_template: dict[str, Any] | None = None,
        response_path: str | None = None,
        context_path: str | None = None,
        tools_path: str | None = None,
        timeout_seconds: float = 60.0,
        method: str = "POST",
    ) -> None:
        self.url = _expand_env(url)
        self.headers = {k: _expand_env(v) for k, v in (headers or {}).items()}
        self.request_template = request_template or {"input": "{input}"}
        self.response_path = response_path
        self.context_path = context_path
        self.tools_path = tools_path
        self.timeout = timeout_seconds
        self.method = method

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
        body = _render(
            self.request_template,
            input=case.input,
            context=case.context,
            model=model,
            provider=provider,
            session_id=session_id,
        )
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    self.method, self.url, json=body, headers=self.headers
                )
        except httpx.TimeoutException as e:
            raise TargetError(f"http target timed out after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise TargetError(f"http target unreachable: {e}") from e
        latency = (time.perf_counter() - start) * 1000
        if response.status_code >= 400:
            raise TargetError(
                f"http target returned {response.status_code}: {response.text[:300]}"
            )
        payload = _parse_response(response)
        return self._to_output(payload, latency_ms=latency)

    def _to_output(self, payload: Any, *, latency_ms: float) -> TargetOutput:
        output = TargetOutput(latency_ms=latency_ms)
        if self.response_path:
            output.output = str(_dig(payload, self.response_path) or "")
        elif isinstance(payload, str):
            output.output = payload
        elif isinstance(payload, dict):
            output.output = str(
                payload.get("output") or payload.get("text") or payload
            )
        if self.context_path:
            context = _dig(payload, self.context_path)
            output.retrieval_context = context if isinstance(context, list) else [context]
        if self.tools_path:
            tools = _dig(payload, self.tools_path)
            output.trajectory = _tools_to_trajectory(tools) if tools else None
        return output


def _render(template: Any, **values: Any) -> Any:
    """Deep-interpolate {placeholders} inside the request template."""
    if isinstance(template, str):
        try:
            return template.format(**{k: v for k, v in values.items() if v is not None})
        except (KeyError, IndexError):
            return template
    if isinstance(template, dict):
        return {key: _render(value, **values) for key, value in template.items()}
    if isinstance(template, list):
        return [_render(item, **values) for item in template]
    return template


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} references — keeps secrets out of suite files."""
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _dig(payload: Any, path: str) -> Any:
    """Dotted-path extraction: 'data.answer' or 'results.0.text'."""
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _parse_response(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.text


def _tools_to_trajectory(tools: Any) -> Any:
    from aitestkit.models import TraceEvent, Trajectory

    if not isinstance(tools, list):
        return None
    events = [
        TraceEvent(type="tool_call", payload={"tool_call": t}, step_index=i)
        for i, t in enumerate(tools)
    ]
    return Trajectory(events=events)
