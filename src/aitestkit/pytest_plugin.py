"""pytest integration: assert_test() helper + suite-driven collection.

Usage in a test::

    from aitestkit import TestCase, assert_test
    from aitestkit.metrics.deterministic import ExactMatch
    from aitestkit.targets.python_target import PythonTarget

    async def test_greeting():
        target = PythonTarget(spec="myapp:answer")
        case = TestCase(input="hi", expected_output="hello")
        output = await target.invoke(case)
        assert_test(case, output, [ExactMatch()])
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
