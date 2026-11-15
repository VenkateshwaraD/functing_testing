"""Data-quality metric — runs expectation specs over case rows.

Case metadata: ``rows`` (list of dicts) and ``expectations`` (spec list).
``backend: gx`` routes through the Great Expectations bridge when the
extra is installed.
"""

from __future__ import annotations

from aitestkit.dataquality.expectations import check_expectations
from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase


@register
class DataQuality(BaseMetric):
    """Evaluate ``metadata['expectations']`` over ``metadata['rows']``."""

    name = "data_quality"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        rows = case.metadata.get("rows")
        specs = case.metadata.get("expectations")
        if not isinstance(rows, list) or not rows:
            return MetricResult.skipped(
                metric=self.name, reason="no rows in case metadata"
            )
        if not isinstance(specs, list) or not specs:
            return MetricResult.skipped(
                metric=self.name, reason="no expectations in case metadata"
            )
        backend = str(case.metadata.get("backend", "builtin"))
        if backend == "gx":
            from aitestkit.dataquality.gx_bridge import run_gx_validation

            report = run_gx_validation(rows, specs)
        else:
            report = check_expectations(rows, specs)
        total = report["total"] or 1
        score = report["passed"] / total
        failures = [r for r in report["results"] if not r["passed"]]
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            reason=(
                "" if not failures
                else "; ".join(
                    f"{f['expectation']}: {f['reason']}" for f in failures[:5]
                )
            ),
            details={
                "backend": report.get("backend", "builtin"),
                "passed": report["passed"],
                "failed": report["failed"],
                "results": report["results"],
            },
        )
