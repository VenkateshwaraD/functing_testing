"""Memory metrics: retention, contradiction, forgetting, accuracy.

Direct inspection of the SUT's ``MemoryStore`` is what separates real
memory testing from re-asking the model. Case metadata drives each check:

- ``must_remember: [facts]`` — context_retention
- ``superseded`` / ``current`` — contradiction_resolution
- ``memory_key`` (optional ``memory_value``) — forgetting_check
- judge-backed memory_accuracy needs no metadata — it reads the store
"""

from __future__ import annotations

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.deterministic import _result
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase


def _session(case: TestCase) -> str | None:
    return case.metadata.get("session_id")


@register
class ContextRetention(BaseMetric):
    """Facts the agent was told must appear in the final-turn answer.

    ``case.metadata['must_remember']``: string or list of strings that the
    last turn's output should still reflect. Score = fraction retained.
    """

    name = "context_retention"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        facts = case.metadata.get("must_remember")
        if not facts:
            return MetricResult.skipped(
                metric=self.name, reason="no must_remember metadata"
            )
        fact_list = facts if isinstance(facts, list) else [facts]
        answer = (output.output or "").lower()
        retained = [f for f in fact_list if str(f).lower() in answer]
        lost = [f for f in fact_list if str(f).lower() not in answer]
        score = len(retained) / len(fact_list)
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"retained {len(retained)}/{len(fact_list)} facts",
            details={"retained": retained, "lost": lost},
        )


@register
class ContradictionResolution(BaseMetric):
    """When facts conflict, the newest must win.

    ``case.metadata['superseded']``: stale fact; ``['current']``: updated
    fact. Score 1.0 when the answer uses current and not superseded.
    """

    name = "contradiction_resolution"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        superseded = case.metadata.get("superseded")
        current = case.metadata.get("current")
        if superseded is None or current is None:
            return MetricResult.skipped(
                metric=self.name, reason="needs superseded+current metadata"
            )
        answer = (output.output or "").lower()
        uses_current = str(current).lower() in answer
        uses_old = str(superseded).lower() in answer
        score = 1.0 if uses_current and not uses_old else 0.0
        if uses_current and uses_old:
            score = 0.5  # mentions both — partial credit
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=(
                "uses updated fact" if score == 1.0
                else "mixes old+new facts" if score == 0.5
                else "uses stale/missing fact"
            ),
            details={
                "superseded": superseded,
                "current": current,
                "used_current": uses_current,
                "used_superseded": uses_old,
            },
        )


@register
class ForgettingCheck(BaseMetric):
    """Direct store inspection: a key must still be present after the run.

    ``case.metadata['memory_key']`` required; optional ``['memory_value']``
    also checks the stored value. Requires ``ctx.memory_store``.
    """

    name = "forgetting_check"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        key = case.metadata.get("memory_key")
        if not key:
            return MetricResult.skipped(
                metric=self.name, reason="no memory_key metadata"
            )
        store = ctx.memory_store
        if store is None:
            return MetricResult.error(
                metric=self.name, reason="no memory_store configured in suite"
            )
        value = store.get(str(key), session_id=_session(case))
        if value is None:
            return _result(
                self.name, score=0.0, threshold=self.threshold,
                reason=f"key {key!r} absent from memory — forgotten",
                details={"memory_key": key},
            )
        expected = case.metadata.get("memory_value")
        if expected is not None and value != expected:
            return _result(
                self.name, score=0.0, threshold=self.threshold,
                reason=f"key {key!r} has wrong value",
                details={"memory_key": key, "expected": expected, "actual": value},
            )
        return _result(
            self.name, score=1.0, threshold=self.threshold,
            reason=f"key {key!r} retained",
            details={"memory_key": key, "value": value},
        )


_MEMORY_PROMPT = """You are evaluating an agent's memory. The agent stored
these facts across a conversation:

STORED FACTS:
{facts}

For the final turn, the agent produced:

ANSWER:
{answer}

Score from 0.0 (answer contradicts or ignores stored facts — memory failure)
to 1.0 (answer is fully consistent with what the agent should remember).
Respond with JSON only: {{"score": <float>, "reason": "<one sentence>"}}
"""


@register
class MemoryAccuracy(BaseMetric):
    """Judge: final answer is consistent with the store's contents."""

    name = "memory_accuracy"
    default_threshold = 0.7

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        store = ctx.memory_store
        if store is None:
            return MetricResult.error(
                metric=self.name, reason="no memory_store configured in suite"
            )
        facts = store.all(session_id=_session(case))
        if not facts:
            return MetricResult.skipped(
                metric=self.name, reason="memory store is empty"
            )
        judge = ctx.require_judge(self.name)
        facts_text = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        verdict, usage = await judge.evaluate(
            _MEMORY_PROMPT.format(facts=facts_text, answer=output.output or "")
        )
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"judge_rationale": verdict.reason, "facts_seen": len(facts)},
            usage=usage,
        )
