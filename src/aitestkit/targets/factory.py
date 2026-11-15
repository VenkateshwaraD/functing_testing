"""Build target adapters from suite target config."""

from __future__ import annotations

from aitestkit.config import TargetConfig
from aitestkit.exceptions import ConfigError
from aitestkit.targets.base import TargetAdapter
from aitestkit.targets.python_target import PythonTarget


def build_target(config: TargetConfig) -> TargetAdapter:
    """Instantiate the adapter described by a target config block."""
    if config.type == "python":
        if not config.function:
            raise ConfigError("python target requires 'function: module:callable'")
        return PythonTarget(spec=config.function, kwargs=config.kwargs)
    raise ConfigError(f"target type {config.type!r} not yet implemented")
