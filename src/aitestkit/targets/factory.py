"""Build target adapters from suite target config.

Dangerous target types require explicit opt-in flags — never enabled by a
YAML file alone, so a fork PR editing a suite cannot gain code execution.
"""

from __future__ import annotations

from aitestkit.config import TargetConfig
from aitestkit.exceptions import ConfigError
from aitestkit.targets.base import TargetAdapter


def build_target(
    config: TargetConfig,
    *,
    allow_python_targets: bool = True,
    allow_command_targets: bool = False,
) -> TargetAdapter:
    """Instantiate the adapter described by a target config block."""
    if config.type == "python":
        if not allow_python_targets:
            raise ConfigError(
                "python targets execute project code — re-run with "
                "--allow-python-targets (default on; disabled here)"
            )
        if not config.function:
            raise ConfigError("python target requires 'function: module:callable'")
        from aitestkit.targets.python_target import PythonTarget

        return PythonTarget(spec=config.function, kwargs=config.kwargs)
    if config.type == "http":
        if not config.url:
            raise ConfigError("http target requires 'url'")
        from aitestkit.targets.http_target import HttpTarget

        return HttpTarget(
            url=config.url,
            headers=config.headers,
            request_template=config.request_template,
            response_path=config.response_path,
            context_path=config.context_path,
            tools_path=config.tools_path,
            timeout_seconds=config.kwargs.get("timeout_seconds", 60.0),
            method=config.kwargs.get("method", "POST"),
        )
    if config.type == "command":
        if not config.command:
            raise ConfigError("command target requires 'command: [argv, ...]'")
        from aitestkit.targets.command_target import CommandTarget

        return CommandTarget(
            command=config.command,
            allowed=allow_command_targets,
            timeout_seconds=config.kwargs.get("timeout_seconds", 120.0),
            env=config.kwargs.get("env"),
        )
    raise ConfigError(f"target type {config.type!r} not implemented")
