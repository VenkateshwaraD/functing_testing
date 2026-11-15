"""Deterministic retrieval metrics against golden labelled ids.

Relevance comes from ``case.expected_retrieval_ids`` vs
``output.retrieval_ids`` (ranked). When ids are unavailable, falls back to
matching retrieved chunk text against ``case.context`` passages — an
approximation, flagged in the result details.
"""

from __future__ import annotations

import math

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase


def _ranked(output: TargetOutput) -> list[str]:
    """Retrieved items in rank order — ids preferred, else chunk text."""
    if output.retrieval_ids:
        return output.retrieval_ids
    return output.retrieval_context or []


def _relevant(case: TestCase) -> list[str] | None:
    """Golden labels. ``expected_retrieval_ids`` is authoritative when set —
    an explicit ``[]`` means 'nothing should be retrieved' (unanswerable
    probe), so it is NOT a skip; it only falls back to ``context`` when the
    field is absent (``None``)."""
    if case.expected_retrieval_ids is not None:
        return case.expected_retrieval_ids
    return case.context if case.context else None


def _matched(ranked: list[str], relevant: list[str]) -> list[int]:
    """Positions (0-based) in ranked list that match a relevant item."""
    rel_set = set(relevant)
    return [i for i, item in enumerate(ranked) if item in rel_set]


class _RankedRetrievalMetric(BaseMetric):
    """Shared plumbing for metrics over ranked retrieval results."""

    def _setup(
        self, case: TestCase, output: TargetOutput
    ) -> tuple[list[str], list[int]] | MetricResult:
        relevant = _relevant(case)
        if relevant is None:
            return MetricResult.skipped(
                metric=self.name,
                reason="no golden relevance labels "
                "(expected_retrieval_ids/context absent)",
            )
        ranked = _ranked(output)
        if not relevant:
            # Explicit 'expect nothing retrieved' — unanswerable probe.
            score = 0.0 if ranked else 1.0
            return MetricResult(
                metric=self.name,
                status=MetricStatus.SCORED,
                score=score,
                passed=score >= self.threshold,
                reason=(
                    "retrieved results despite empty golden labels"
                    if ranked
                    else "correct abstention — nothing retrieved"
                ),
                details={"retrieved": ranked, "expected_labels": 0},
            )
        if not ranked:
            return MetricResult(
                metric=self.name,
                status=MetricStatus.SCORED,
                score=0.0,
                passed=self.threshold <= 0.0,
                reason="expected retrieval but nothing was retrieved",
                details={"expected_labels": relevant},
            )
        return ranked, _matched(ranked, relevant)


@register
class RecallAtK(_RankedRetrievalMetric):
    """|relevant ∩ retrieved[:k]| / |relevant|."""

    name = "recall_at_k"
    default_threshold = 0.5

    def __init__(self, *, k: int = 5, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.k = k

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        setup = self._setup(case, output)
        if isinstance(setup, MetricResult):
            return setup
        _, matched = setup
        relevant = _relevant(case) or []
        hits = sum(1 for pos in matched if pos < self.k)
        score = hits / len(relevant)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={
                "k": self.k,
                "relevant": len(relevant),
                "hits_at_k": hits,
                "approximate": not case.expected_retrieval_ids,
            },
        )


@register
class PrecisionAtK(_RankedRetrievalMetric):
    """|relevant ∩ retrieved[:k]| / k."""

    name = "precision_at_k"
    default_threshold = 0.5

    def __init__(self, *, k: int = 5, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.k = k

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        setup = self._setup(case, output)
        if isinstance(setup, MetricResult):
            return setup
        _, matched = setup
        hits = sum(1 for pos in matched if pos < self.k)
        score = hits / self.k
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={"k": self.k, "hits_at_k": hits},
        )


@register
class MRR(_RankedRetrievalMetric):
    """Reciprocal rank of the first relevant retrieval."""

    name = "mrr"
    default_threshold = 0.5

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        setup = self._setup(case, output)
        if isinstance(setup, MetricResult):
            return setup
        _, matched = setup
        score = 1.0 / (matched[0] + 1) if matched else 0.0
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={
                "first_relevant_rank": matched[0] + 1 if matched else None
            },
        )


@register
class NDCG(_RankedRetrievalMetric):
    """Normalized discounted cumulative gain over the ranked list."""

    name = "ndcg"
    default_threshold = 0.5

    def __init__(self, *, k: int = 10, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.k = k

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        setup = self._setup(case, output)
        if isinstance(setup, MetricResult):
            return setup
        _, matched = setup
        relevant = _relevant(case) or []
        dcg = sum(
            1.0 / math.log2(pos + 2) for pos in matched if pos < self.k
        )
        ideal_hits = min(len(relevant), self.k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        score = dcg / idcg if idcg else 0.0
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={"k": self.k, "dcg": dcg, "idcg": idcg},
        )


@register
class HitRate(_RankedRetrievalMetric):
    """1.0 if any relevant item appears in top-k."""

    name = "hit_rate"
    default_threshold = 1.0

    def __init__(self, *, k: int = 5, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.k = k

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        setup = self._setup(case, output)
        if isinstance(setup, MetricResult):
            return setup
        _, matched = setup
        hit = any(pos < self.k for pos in matched)
        score = 1.0 if hit else 0.0
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={"k": self.k},
        )


@register
class RerankDelta(_RankedRetrievalMetric):
    """Improvement in nDCG after reranking.

    Reads ``output.metadata['pre_rerank_ids']`` (ranked ids before rerank)
    vs ``output.retrieval_ids`` (after). Positive = reranker helped.
    """

    name = "rerank_delta"
    default_threshold = 0.0

    def __init__(self, *, k: int = 10, threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.k = k

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        pre = output.metadata.get("pre_rerank_ids")
        post = output.retrieval_ids
        relevant = _relevant(case)
        if not post or not relevant:
            return MetricResult.skipped(
                metric=self.name,
                reason="missing retrieval_ids or golden labels",
            )
        if not pre:
            return MetricResult.skipped(
                metric=self.name,
                reason="output.metadata['pre_rerank_ids'] absent",
            )
        rel = set(relevant)

        def ndcg(ranked: list[str]) -> float:
            dcg = sum(
                1.0 / math.log2(i + 2)
                for i, item in enumerate(ranked[: self.k])
                if item in rel
            )
            ideal = min(len(rel), self.k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
            return dcg / idcg if idcg else 0.0

        before, after = ndcg(pre), ndcg(post)
        delta = after - before
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=max(0.0, min(1.0, 0.5 + delta)),  # map [-1,1]→[0,1]
            passed=delta >= self.threshold,
            details={
                "k": self.k,
                "ndcg_before": before,
                "ndcg_after": after,
                "delta": delta,
            },
        )
