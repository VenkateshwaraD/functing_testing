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
    register_builtins()
    return sorted(_REGISTRY)


def metric_exists(name: str) -> bool:
    """True when a metric is registered under ``name``."""
    register_builtins()
    return name in _REGISTRY


def metric_requires_judge(name: str) -> bool:
    """True when the metric needs an LLM judge to run."""
    register_builtins()
    cls = _REGISTRY.get(name)
    return bool(cls and cls.requires_judge)


def register_builtins() -> None:
    """Import every built-in metric module so decorators fire.

    Idempotent — imports are cheap after the first call. Anything that
    resolves metric names (CLI, pytest plugin, library users) should call
    this first.
    """
    import aitestkit.dataquality.metric  # noqa: F401
    import aitestkit.metrics.agentic  # noqa: F401
    import aitestkit.metrics.backends.deepeval_backend  # noqa: F401
    import aitestkit.metrics.backends.ragas_backend  # noqa: F401
    import aitestkit.metrics.deterministic  # noqa: F401
    import aitestkit.metrics.ingestion  # noqa: F401
    import aitestkit.metrics.llm_judge  # noqa: F401
    import aitestkit.metrics.memory  # noqa: F401
    import aitestkit.metrics.prompts  # noqa: F401
    import aitestkit.metrics.rag  # noqa: F401
    import aitestkit.metrics.retrieval  # noqa: F401
    import aitestkit.vectordb.checks  # noqa: F401


def build_metric(name: str, *, threshold: float | None, params: dict[str, Any]) -> BaseMetric:
    """Instantiate a registered metric from a suite spec."""
    register_builtins()
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
