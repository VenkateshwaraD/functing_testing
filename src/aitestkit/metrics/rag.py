"""RAG generation-stage metrics.

Faithfulness uses the claim-decomposition algorithm (RAGAS-parity):
decompose the answer into atomic claims, verify each against the retrieved
context, score = supported/total — implemented as one structured judge call
whose JSON carries per-claim verdicts into ``details`` for debuggability.
"""

from __future__ import annotations

import re

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase, Usage
from aitestkit.rag.token_counting import count_tokens


def _judge_payload(raw: dict) -> dict:
    """Unwrap self-consistency samples to the first raw judge payload."""
    samples = raw.get("samples")
    if isinstance(samples, list) and samples and isinstance(samples[0], dict):
        return samples[0]
    return raw


def _context_text(case: TestCase, output: TargetOutput) -> str:
    chunks = output.retrieval_context or case.context or []
    return "\n---\n".join(chunks)


@register
class Faithfulness(BaseMetric):
    """Claim decomposition → per-claim verification → supported/total.

    The judge returns ``{"claims": [{"claim", "supported", "evidence"}],
    "score", "reason"}``; claims land in details so failures name exactly
    which statements were unsupported.
    """

    name = "faithfulness"
    requires_judge = True
    required_fields = ()
    default_threshold = 0.85

    _PROMPT = """\
You are evaluating whether an AI answer is faithful to the given context.

Context:
{context}

Answer:
{answer}

Steps:
1. Decompose the answer into atomic factual claims.
2. For each claim, judge whether it is directly supported by the context.
3. Report score = supported_claims / total_claims (1.0 if no claims).

Return JSON:
{{"claims": [{{"claim": "...", "supported": true, "evidence": "..."}}],
  "score": 0.0-1.0, "reason": "brief overall justification"}}"""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        context = _context_text(case, output)
        if not context:
            return MetricResult.skipped(
                metric=self.name, reason="no context or retrieval_context"
            )
        verdict, usage = await judge.evaluate(
            self._PROMPT.format(context=context, answer=output.output or "")
        )
        claims = _judge_payload(verdict.raw).get("claims")
        if isinstance(claims, list) and claims:
            supported = [c for c in claims if isinstance(c, dict) and c.get("supported")]
            score = len(supported) / len(claims)
        else:
            score = verdict.score
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            reason=verdict.reason,
            details={
                "claims": claims if isinstance(claims, list) else [],
                "unsupported": [
                    c.get("claim", "")
                    for c in (claims or [])
                    if isinstance(c, dict) and not c.get("supported")
                ],
            },
            usage=usage,
        )


@register
class AnswerRelevancy(BaseMetric):
    """Judge: does the answer address the question (non-committal check)."""

    name = "answer_relevancy"
    requires_judge = True
    default_threshold = 0.7

    _PROMPT = """\
Question: {question}
Answer: {answer}

Score how directly the answer addresses the question. Penalize
non-committal answers ("I don't know", deflection) when the question is
answerable, and tangential content.

Return JSON: {{"score": 0.0-1.0, "reason": "..."}}"""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        verdict, usage = await judge.evaluate(
            self._PROMPT.format(
                question=case.input, answer=output.output or ""
            )
        )
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"rationale": verdict.reason},
            usage=usage,
        )


@register
class Groundedness(Faithfulness):
    """Statement-level variant — verifies no claim goes beyond context."""

    name = "groundedness"
    default_threshold = 0.85

    _PROMPT = """\
You are checking whether every statement in an answer is grounded in the
provided context (no hallucination beyond it).

Context:
{context}

Answer:
{answer}

Steps:
1. List each factual statement in the answer.
2. For each, decide whether the context supports it verbatim or by
   reasonable paraphrase.
3. score = supported / total.

Return JSON:
{{"claims": [{{"claim": "...", "supported": true, "evidence": "..."}}],
  "score": 0.0-1.0, "reason": "..."}}"""


@register
class ContextualRecall(BaseMetric):
    """Judge: can the expected answer be derived from the retrieved context?

    Attribution-friendly: a low score here means retrieval failed, not the
    generator.
    """

    name = "contextual_recall"
    requires_judge = True
    required_fields = ("expected_output",)
    default_threshold = 0.7

    _PROMPT = """\
Retrieved context:
{context}

Expected answer:
{expected}

Does the context contain all information needed to produce the expected
answer? Identify which facts are missing.

Return JSON: {{"score": 0.0-1.0, "reason": "...",
"missing_facts": ["..."]}}"""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        context = _context_text(case, output)
        if not context:
            return MetricResult.skipped(
                metric=self.name, reason="no context or retrieval_context"
            )
        verdict, usage = await judge.evaluate(
            self._PROMPT.format(
                context=context, expected=case.expected_output or ""
            )
        )
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={
                "missing_facts": _judge_payload(verdict.raw).get("missing_facts", []),
                "attribution_hint": "retrieval",
            },
            usage=usage,
        )


@register
class ContextualPrecision(BaseMetric):
    """Judge: are the retrieved chunks relevant to the question?

    Reports a rank-order aware verdict — irrelevant chunks ranked high are
    the reranking/retrieval problem this metric isolates.
    """

    name = "contextual_precision"
    requires_judge = True
    default_threshold = 0.7

    _PROMPT = """\
Question: {question}

Retrieved chunks (in rank order):
{context}

For each chunk decide if it is relevant to answering the question.
score = relevant / total.

Return JSON: {{"score": 0.0-1.0, "reason": "...",
"irrelevant_ranks": [1, 4]}}"""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        context = _context_text(case, output)
        if not context:
            return MetricResult.skipped(
                metric=self.name, reason="no context or retrieval_context"
            )
        verdict, usage = await judge.evaluate(
            self._PROMPT.format(question=case.input, context=context)
        )
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={
                "irrelevant_ranks": _judge_payload(verdict.raw).get("irrelevant_ranks", []),
                "attribution_hint": "retrieval",
            },
            usage=usage,
        )


_CITATION_RE = re.compile(r"\[(\d+)\]|\[\^(\w+)\]|\((\d+)\)")


@register
class CitationAccuracy(BaseMetric):
    """Each citation in the answer must map to a supporting context chunk.

    Citations match ``[N]``, ``[^label]``, ``(N)`` against 1-indexed
    retrieval_context positions.
    """

    name = "citation_accuracy"
    requires_judge = True
    default_threshold = 0.8

    _PROMPT = """\
Context chunk {idx}: {chunk}
Cited claim: {claim}

Does this chunk directly support the claim?
Return JSON: {{"score": 1.0 or 0.0, "reason": "..."}}"""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        text = output.output or ""
        chunks = output.retrieval_context or case.context or []
        matches = [m for m in _CITATION_RE.finditer(text)]
        if not matches:
            return MetricResult.skipped(
                metric=self.name, reason="no citations in answer"
            )
        verified = 0
        total = len(matches)
        usage_total = Usage()
        details: list[dict[str, object]] = []
        for match in matches:
            ref = next(g for g in match.groups() if g)
            try:
                idx = int(ref) - 1
            except ValueError:
                idx = -1
            ok = 0 <= idx < len(chunks)
            claim = text[max(0, match.start() - 120) : match.start()].strip()
            if ok:
                verdict, usage = await judge.evaluate(
                    self._PROMPT.format(
                        idx=idx + 1, chunk=chunks[idx], claim=claim
                    )
                )
                usage_total += usage
                ok = ok and verdict.score >= 0.5
                details.append(
                    {"ref": ref, "supported": ok, "reason": verdict.reason}
                )
            else:
                details.append({"ref": ref, "supported": False, "reason": "no such chunk"})
            verified += int(ok)
        score = verified / total
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={"citations": details, "verified": verified, "total": total},
            usage=usage_total,
        )


@register
class ContextWindow(BaseMetric):
    """Deterministic: input + context must fit the model's token budget.

    Uses vendored tiktoken when available; approximate otherwise. Fails
    (not errors) when over budget — that's a quality/config problem the
    suite should catch.
    """

    name = "context_window"
    default_threshold = 1.0

    def __init__(
        self, *, max_tokens: int = 128000, threshold: float | None = None
    ) -> None:
        super().__init__(threshold=threshold)
        self.max_tokens = max_tokens

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        context = _context_text(case, output)
        used = count_tokens(case.input) + count_tokens(context)
        headroom = self.max_tokens - used
        passed = headroom >= 0
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=(
                "" if passed
                else f"context overflows budget by {-headroom} tokens"
            ),
            details={
                "tokens_used": used,
                "max_tokens": self.max_tokens,
                "headroom": headroom,
                "approximate": True,
            },
        )


@register
class MultiTurnCoherence(BaseMetric):
    """Judge: consistency of the final answer with earlier turns.

    Reads ``case.metadata['turn_outputs']`` — the conversation runner
    injects prior (input, output) pairs there for the last turn.
    """

    name = "multi_turn_coherence"
    requires_judge = True
    default_threshold = 0.7

    _PROMPT = """\
Conversation so far:
{history}

Final answer: {answer}

Is the final answer consistent with the conversation — no contradictions,
correct carry-over of established facts?
Return JSON: {{"score": 0.0-1.0, "reason": "...",
"contradictions": ["..."]}}"""

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        history = case.metadata.get("turn_outputs") or []
        if not history:
            return MetricResult.skipped(
                metric=self.name, reason="no turn_outputs in metadata"
            )
        rendered = "\n".join(
            f"User: {t.get('input', '')}\nAssistant: {t.get('output', '')}"
            for t in history
        )
        verdict, usage = await judge.evaluate(
            self._PROMPT.format(history=rendered, answer=output.output or "")
        )
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"contradictions": _judge_payload(verdict.raw).get("contradictions", [])},
            usage=usage,
        )
