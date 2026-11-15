"""Deterministic metrics: no model calls, instant, fully reproducible."""

from __future__ import annotations

import re
import time
from typing import Any

from jsonschema import ValidationError, validate

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase


def _result(
    name: str,
    *,
    score: float,
    threshold: float,
    reason: str,
    details: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
) -> MetricResult:
    return MetricResult(
        metric=name,
        status=MetricStatus.SCORED,
        score=score,
        passed=score >= threshold,
        reason=reason,
        details=details or {},
        duration_ms=duration_ms,
    )


@register
class ExactMatch(BaseMetric):
    """Pass when actual output equals expected output (optionally normalized)."""

    name = "exact_match"
    required_fields = ("expected_output",)
    default_threshold = 1.0

    def __init__(self, *, threshold: float | None = None, normalize: bool = True) -> None:
        super().__init__(threshold=threshold)
        self.normalize = normalize

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        actual = output.output or ""
        expected = case.expected_output or ""
        if self.normalize:
            actual, expected = actual.strip(), expected.strip()
        score = 1.0 if actual == expected else 0.0
        reason = "exact match" if score else "output differs from expected"
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=reason,
            details={"expected": expected, "actual": actual},
        )


@register
class Contains(BaseMetric):
    """Pass when the output contains all (or any) required substrings."""

    name = "contains"
    default_threshold = 1.0

    def __init__(
        self,
        *,
        substrings: list[str] | None = None,
        any_of: bool = False,
        threshold: float | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.substrings = substrings or []
        self.any_of = any_of

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        haystack = output.output or ""
        hits = [s for s in self.substrings if s in haystack]
        missing = [s for s in self.substrings if s not in haystack]
        if self.any_of:
            score = 1.0 if hits else 0.0
        else:
            score = len(hits) / len(self.substrings) if self.substrings else 1.0
        reason = f"{len(hits)}/{len(self.substrings)} substrings present"
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=reason,
            details={"found": hits, "missing": missing},
        )


@register
class RegexMatch(BaseMetric):
    """Pass when the output matches a regex pattern."""

    name = "regex_match"
    default_threshold = 1.0

    def __init__(self, *, pattern: str, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.pattern = pattern

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        text = output.output or ""
        match = re.search(self.pattern, text, flags=re.DOTALL)
        score = 1.0 if match else 0.0
        reason = f"pattern {'matched' if match else 'not found'}"
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=reason,
            details={"pattern": self.pattern, "match": match.group(0) if match else None},
        )


@register
class JsonSchema(BaseMetric):
    """Validate that the output parses as JSON matching a JSON Schema."""

    name = "json_schema"
    default_threshold = 1.0

    def __init__(self, *, schema: dict[str, Any], threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.schema = schema

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        import json

        try:
            payload = json.loads(output.output or "")
        except json.JSONDecodeError as e:
            return _result(
                self.name,
                score=0.0,
                threshold=self.threshold,
                reason=f"output is not valid JSON: {e}",
            )
        try:
            validate(instance=payload, schema=self.schema)
        except ValidationError as e:
            return _result(
                self.name,
                score=0.0,
                threshold=self.threshold,
                reason=f"schema violation: {e.message}",
                details={"path": list(e.absolute_path)},
            )
        return _result(
            self.name, score=1.0, threshold=self.threshold, reason="valid against schema"
        )


@register
class Latency(BaseMetric):
    """Pass when target latency is under a bound."""

    name = "latency"
    default_threshold = 1.0

    def __init__(self, *, max_ms: float, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.max_ms = max_ms

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        score = 1.0 if output.latency_ms <= self.max_ms else 0.0
        reason = f"{output.latency_ms:.0f}ms vs max {self.max_ms:.0f}ms"
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=reason,
            details={"latency_ms": output.latency_ms, "max_ms": self.max_ms},
        )


@register
class MaxCost(BaseMetric):
    """Pass when the SUT call cost is under a per-case bound."""

    name = "max_cost"
    default_threshold = 1.0

    def __init__(self, *, max_usd: float, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.max_usd = max_usd

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        cost = output.usage.cost_usd
        score = 1.0 if cost <= self.max_usd else 0.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"${cost:.4f} vs max ${self.max_usd:.4f}",
            details={"cost_usd": cost, "max_usd": self.max_usd},
        )


class _TimedMetric(BaseMetric):
    """Helper: wrap measure in a duration capture for judge metrics."""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        start = time.perf_counter()
        result = await self._timed(case, output, ctx)
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    async def _timed(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        raise NotImplementedError
