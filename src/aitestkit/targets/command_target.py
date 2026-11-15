"""Command target: run an argv list as the system under test.

Never ``shell=True`` — the command is an explicit argv list. The case input
goes to stdin as JSON; stdout is parsed as JSON (``output``/``trajectory``/
``usage`` fields) or returned as raw text. The process group is killed on
timeout — this is also the isolation path for stuck synchronous agents.
Requires explicit opt-in via ``allowed=True`` (the CLI passes
``--allow-command-targets``); a YAML file alone cannot enable it.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from aitestkit.exceptions import ConfigError, TargetError
from aitestkit.models import TargetOutput, TestCase, Trajectory, Usage
from aitestkit.targets.base import child_env


class CommandTarget:
    """Run ``argv`` per case; killable, shell-free, opt-in only."""

    name = "command"

    def __init__(
        self,
        *,
        command: list[str],
        allowed: bool,
        timeout_seconds: float = 120.0,
        env: dict[str, str] | None = None,
    ) -> None:
        if not allowed:
            raise ConfigError(
                "command targets run arbitrary code — re-run with "
                "--allow-command-targets (or AITESTKIT_ALLOW_COMMAND_TARGETS=1)"
            )
        if not command:
            raise ConfigError("command target requires a non-empty argv list")
        self.command = command
        self.timeout = timeout_seconds
        self.env = env

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
        payload = json.dumps(
            {
                "input": case.input,
                "context": case.context,
                "expected_output": case.expected_output,
                "model": model,
                "provider": provider,
                "session_id": session_id,
                "case_id": case.id,
            }
        )
        start = time.perf_counter()
        stdout, stderr = await self._run_process(payload)
        latency = (time.perf_counter() - start) * 1000
        return _parse_stdout(stdout, stderr=stderr, latency_ms=latency)

    async def _run_process(self, stdin_payload: str) -> tuple[str, str]:
        env = child_env(self.env)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            raise TargetError(f"command not found: {self.command[0]}") from e
        try:
            out_bytes, err_bytes = await asyncio.wait_for(
                process.communicate(stdin_payload.encode()), timeout=self.timeout
            )
        except TimeoutError as e:
            process.kill()
            await process.wait()
            raise TargetError(
                f"command target killed after {self.timeout}s timeout"
            ) from e
        stdout = out_bytes.decode(errors="replace")
        stderr = err_bytes.decode(errors="replace")
        if process.returncode != 0:
            raise TargetError(
                f"command exited {process.returncode}: {stderr[:500] or stdout[:500]}"
            )
        return stdout, stderr


def _parse_stdout(stdout: str, *, stderr: str, latency_ms: float) -> TargetOutput:
    """JSON stdout -> structured TargetOutput; otherwise raw text."""
    text = stdout.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return TargetOutput(output=text, latency_ms=latency_ms)
    if not isinstance(payload, dict):
        return TargetOutput(output=text, latency_ms=latency_ms)
    trajectory = payload.get("trajectory")
    return TargetOutput(
        output=payload.get("output") or payload.get("text") or text,
        retrieval_context=payload.get("retrieval_context") or payload.get("context"),
        trajectory=Trajectory.model_validate(trajectory) if trajectory else None,
        usage=Usage.model_validate(payload.get("usage") or {}),
        latency_ms=latency_ms,
    )
