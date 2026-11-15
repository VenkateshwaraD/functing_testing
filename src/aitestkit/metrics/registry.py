"""Metric registry: resolve suite metric specs to metric instances."""

from __future__ import annotations

from typing import Any

from aitestkit.exceptions import ConfigError
from aitestkit.metrics.base import BaseMetric

_REGISTRY: dict[str, type[BaseMetric]] = {}


def register(cls: type[BaseMetric]) -> type[BaseMetric]:
    """Class decorator registering a metric under its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def metric_names() -> list[str]:
    """All registered metric names."""
    return sorted(_REGISTRY)


def build_metric(name: str, *, threshold: float | None, params: dict[str, Any]) -> BaseMetric:
    """Instantiate a registered metric from a suite spec."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ConfigError(
            f"unknown metric {name!r}. Available: {', '.join(metric_names())}"
        )
    kwargs = dict(params)
    if threshold is not None:
        kwargs["threshold"] = threshold
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"metric {name!r}: bad params {params}: {e}") from e
