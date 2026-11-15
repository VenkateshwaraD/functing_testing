"""MCP target: call a tool on a Model Context Protocol server as the SUT.

``type: mcp`` — ``command: [argv]`` spawns a stdio server subprocess (same
risk class as ``type: command``, so it needs ``--allow-command-targets``);
``url:`` connects to a remote server over streamable HTTP. ``kwargs.tool``
names the tool invoked per case; ``kwargs.args_template`` maps tool
arguments with ``{input}``/``{session_id}`` substitution.

Each call opens a fresh client session — MCP server-side state does not
persist across cases. Requires the ``mcp`` extra:
``pip install 'aitestkit[mcp]'``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from aitestkit.exceptions import ConfigError, TargetError
from aitestkit.models import TargetOutput, TestCase, ToolCall


class _NullSink:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def _describe(command: list[str] | None, url: str | None) -> str:
    """Log-safe target descriptor — strips URL userinfo so credentials
    never land in the trajectory."""
    if command:
        return command[0]
    return (url or "").rsplit("@", 1)[-1]


def _client_session() -> Any:
    try:
        from mcp import ClientSession
    except ImportError as e:
        raise ConfigError(
            "mcp target requires the MCP SDK — pip install 'aitestkit[mcp]'"
        ) from e
    return ClientSession


def _http_client() -> Any:
    """streamable-HTTP client — ``streamable_http_client`` on mcp>=2,
    ``streamablehttp_client`` on mcp 1.x."""
    try:
        import mcp.client.streamable_http as streamable_http
    except ImportError as e:
        raise ConfigError(
            "mcp target url: requires streamable-HTTP support in the "
            "installed MCP SDK"
        ) from e
    client = getattr(streamable_http, "streamable_http_client", None) or getattr(
        streamable_http, "streamablehttp_client", None
    )
    if client is None:
        raise ConfigError(
            "mcp target url: no streamable-HTTP client in the installed MCP SDK"
        )
    return client


class McpTarget:
    """Invoke one configured tool per case over stdio or streamable HTTP."""

    name = "mcp"

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        allowed: bool,
        tool: str | None,
        args_template: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
        env: dict[str, str] | None = None,
    ) -> None:
        if command and url:
            raise ConfigError("mcp target takes 'command' or 'url', not both")
        if not command and not url:
            raise ConfigError(
                "mcp target requires 'command: [argv, ...]' or 'url:'"
            )
        if command and not allowed:
            raise ConfigError(
                "mcp stdio targets spawn a server process — re-run with "
                "--allow-command-targets (or AITESTKIT_ALLOW_COMMAND_TARGETS=1)"
            )
        if not tool:
            raise ConfigError(
                "mcp target requires 'kwargs.tool' — the server tool to call per case"
            )
        self.command = command
        self.url = url
        self.tool = tool
        self.args_template = args_template or {"input": "{input}"}
        self.timeout = timeout_seconds
        self.env = env
        self._desc = _describe(command, url)

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[Any]:
        client_session = _client_session()
        if self.command:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self.command[0], args=self.command[1:], env=self.env
            )
            async with stdio_client(params) as (read, write), client_session(
                read, write
            ) as session:
                yield session
        else:
            http_client = _http_client()
            async with http_client(self.url) as (read, write, _sid), client_session(
                read, write
            ) as session:
                yield session

    def _render_args(self, case: TestCase, session_id: str | None) -> dict[str, Any]:
        mapping = {
            "{input}": case.input,
            "{session_id}": session_id or "",
            "{case_id}": case.id or "",
        }
        args: dict[str, Any] = {}
        for key, value in self.args_template.items():
            if isinstance(value, str):
                args[key] = mapping.get(value, value)
            else:
                args[key] = value
        return args

    async def _call(self, args: dict[str, Any]) -> Any:
        async with self._connect() as session:
            await session.initialize()
            return await session.call_tool(self.tool, args)

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
        sink = trace_sink if trace_sink is not None else _NullSink()
        args = self._render_args(case, session_id)
        sink.thought(f"mcp: {self.tool}({args}) via {self._desc}")
        try:
            result = await asyncio.wait_for(self._call(args), timeout=self.timeout)
        except TimeoutError as e:
            raise TargetError(
                f"mcp target timed out after {self.timeout}s"
            ) from e
        except (ConfigError, TargetError):
            raise
        except Exception as e:  # noqa: BLE001 — surfaced as TargetError
            raise TargetError(f"mcp target {self._desc} raised: {e}") from e

        output_text, error = self._shape_result(result)
        call = ToolCall(
            name=self.tool,
            args=args,
            result=output_text or None,
            error=error,
        )
        sink.tool_call(self.tool, args=args, result=output_text, error=error)
        sink.message("assistant", output_text)
        sink.finish("error" if error else "done")
        return TargetOutput(output=output_text, tool_calls=[call])

    @staticmethod
    def _shape_result(result: Any) -> tuple[str, str | None]:
        """Extract text + error flag from a CallToolResult."""
        parts = [
            c.text
            for c in getattr(result, "content", [])
            if getattr(c, "type", None) == "text" and getattr(c, "text", None)
        ]
        output_text = "\n".join(parts)
        structured = getattr(result, "structuredContent", None)
        if not output_text and structured is not None:
            output_text = json.dumps(structured)
        error = (
            "tool call returned isError"
            if getattr(result, "isError", False)
            else None
        )
        return output_text, error
