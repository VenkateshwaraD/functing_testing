"""Metric backend tests: deepeval/ragas adapters without the extras."""

from __future__ import annotations

from aitestkit.metrics.backends.deepeval_backend import DeepEvalMetric
from aitestkit.metrics.backends.ragas_backend import RagasMetric
from aitestkit.metrics.base import MetricContext
from aitestkit.models import MetricStatus, TargetOutput, TestCase


async def test_deepeval_missing_extra_is_error_not_crash() -> None:
    try:
        import deepeval  # noqa: F401

        return  # extra installed — backend would run for real
    except ImportError:
        pass
    result = await DeepEvalMetric(metric="faithfulness").measure(
        TestCase(id="t", input="q", context=["ctx"]),
        TargetOutput(output="ans"),
        MetricContext(),
    )
    assert result.status == MetricStatus.ERROR
    assert "aitestkit[deepeval]" in result.reason


async def test_ragas_missing_extra_is_error_not_crash() -> None:
    try:
        import ragas  # noqa: F401

        return
    except ImportError:
        pass
    result = await RagasMetric(metric="faithfulness").measure(
        TestCase(id="t", input="q", context=["ctx"]),
        TargetOutput(output="ans"),
        MetricContext(),
    )
    assert result.status == MetricStatus.ERROR
    assert "aitestkit[ragas]" in result.reason


def test_backends_registered() -> None:
    from aitestkit.metrics import registry

    assert "deepeval" in registry.metric_names()
    assert "ragas" in registry.metric_names()
