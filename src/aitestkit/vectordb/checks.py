"""Vector DB checks — metrics that inspect the store via MetricContext.

``vectordb_count`` verifies the index holds the expected document count;
``embedding_drift`` re-embeds a sample and measures cosine shift against
stored vectors — the check that catches an embedder or chunker change
breaking an existing index.
"""

from __future__ import annotations

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase


@register
class VectorDbCount(BaseMetric):
    """Index must contain at least ``min_count`` (or exactly ``expected``)."""

    name = "vectordb_count"
    default_threshold = 1.0

    def __init__(
        self,
        *,
        min_count: int = 1,
        expected: int | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.min_count = min_count
        self.expected = expected

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        store = ctx.vector_store
        if store is None:
            return MetricResult.error(
                metric=self.name, reason="no vector store in metric context"
            )
        count = await store.count()
        if self.expected is not None:
            passed = count == self.expected
            reason = "" if passed else f"expected {self.expected}, got {count}"
        else:
            passed = count >= self.min_count
            reason = "" if passed else f"only {count} docs (min {self.min_count})"
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=reason,
            details={"count": count, "min_count": self.min_count},
        )


@register
class EmbeddingDrift(BaseMetric):
    """Cosine shift between stored vectors and fresh embeddings.

    Score = mean stored-vs-fresh similarity (1.0 = no drift). A changed
    embedder or chunker drops this — exactly the regression it's for.
    """

    name = "embedding_drift"
    default_threshold = 0.95

    def __init__(
        self,
        *,
        sample_fraction: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.sample_fraction = sample_fraction

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        store = ctx.vector_store
        if store is None:
            return MetricResult.error(
                metric=self.name, reason="no vector store in metric context"
            )
        try:
            sample = await store.reembed_sample(fraction=self.sample_fraction)
        except Exception as e:  # noqa: BLE001 — remote stores raise ConfigError
            return MetricResult.error(metric=self.name, reason=str(e))
        if not sample:
            return MetricResult.skipped(
                metric=self.name, reason="store is empty — nothing to drift-check"
            )
        sims = [sim for _, sim in sample]
        mean_sim = sum(sims) / len(sims)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=mean_sim,
            passed=mean_sim >= self.threshold,
            reason=(
                "" if mean_sim >= self.threshold
                else f"mean cosine similarity {mean_sim:.3f} below {self.threshold}"
            ),
            details={
                "sampled": len(sample),
                "mean_similarity": mean_sim,
                "min_similarity": min(sims),
                "worst_chunk": min(sample, key=lambda p: p[1])[0].id,
            },
        )
