"""Imperative test helper — usable from pytest or any async test runner.

Kept free of pytest imports so the base package (and the CLI) works in
environments without the dev extras installed.
"""

from __future__ import annotations

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.models import MetricResult, TargetOutput, TestCase


async def assert_test(
    case: TestCase,
    output: TargetOutput,
    metrics: list[BaseMetric],
    *,
    ctx: MetricContext | None = None,
) -> list[MetricResult]:
    """Evaluate metrics on a case output; raise AssertionError on failure."""
    context = ctx or MetricContext()
    results = [await metric.evaluate(case, output, context) for metric in metrics]
    failures = [
        f"{r.metric} score={r.score} reason={r.reason}"
        for r in results
        if r.passed is False
    ]
    errors = [
        f"{r.metric} error={r.reason}" for r in results if r.status.value == "error"
    ]
    problems = failures + errors
    if problems:
        raise AssertionError(
            f"aitestkit case {case.id or '<unnamed>'} failed:\n  " + "\n  ".join(problems)
        )
    return results
