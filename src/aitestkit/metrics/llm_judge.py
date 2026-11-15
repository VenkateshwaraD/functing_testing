"""Judge-backed metrics: correctness vs expected output and custom rubrics."""

from __future__ import annotations

import json
from typing import Any

from aitestkit.judge.judge import wrap_untrusted
from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase

_CORRECTNESS_PROMPT = """You are an impartial evaluator. Compare the ACTUAL output
to the EXPECTED output for the given INPUT. Score semantic correctness from
0.0 (completely wrong) to 1.0 (fully correct), ignoring phrasing differences.

INPUT:
{input}

EXPECTED OUTPUT:
{expected}

ACTUAL OUTPUT:
{actual}

Respond with JSON only: {{"score": <float>, "reason": "<one sentence>",
"details": {{"missing": ["facts expected but absent"],
"incorrect": ["facts present but wrong"]}}}}
"""

_RUBRIC_PROMPT = """You are an impartial evaluator. Score the OUTPUT for the
INPUT against this rubric:

RUBRIC: {rubric}
CRITERIA: {criteria}

INPUT:
{input}

OUTPUT:
{actual}

Respond with JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
"""


@register
class Correctness(BaseMetric):
    """Judge-scored semantic correctness against ``expected_output``."""

    name = "correctness"
    requires_judge = True
    required_fields = ("expected_output",)
    default_threshold = 0.7

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        prompt = _CORRECTNESS_PROMPT.format(
            input=case.input,
            expected=case.expected_output,
            actual=wrap_untrusted(output.output or "", label="ACTUAL OUTPUT"),
        )
        verdict, usage = await judge.evaluate(prompt)
        details: dict[str, Any] = {"judge_rationale": verdict.reason}
        details.update(verdict.raw.get("details") or {})
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details=details,
            usage=usage,
        )


@register
class CustomRubric(BaseMetric):
    """G-Eval-style metric: score an output against a user rubric/criteria."""

    name = "custom_rubric"
    requires_judge = True
    default_threshold = 0.7

    def __init__(
        self,
        *,
        rubric: str,
        criteria: str = "",
        threshold: float | None = None,
        metric_name: str | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.rubric = rubric
        self.criteria = criteria
        self.name = metric_name or self.name

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        prompt = _RUBRIC_PROMPT.format(
            rubric=self.rubric,
            criteria=self.criteria or "none",
            input=case.input,
            actual=wrap_untrusted(output.output or "", label="OUTPUT"),
        )
        verdict, usage = await judge.evaluate(prompt)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"judge_rationale": verdict.reason, "rubric": self.rubric},
            usage=usage,
        )


def parse_claims(text: str) -> list[str]:
    """Parse a judge claim-decomposition response into claim strings."""
    try:
        payload = json.loads(text)
        claims = payload.get("claims", [])
        return [str(c) for c in claims if c]
    except (json.JSONDecodeError, AttributeError):
        return [line.strip("- ") for line in text.splitlines() if line.strip()]
