"""Metric contract: every check produces a MetricResult.

Metrics declare which TestCase fields they require; cases missing them get
``skipped``, not ``failed``. LLM metrics take a Judge from the context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase

if TYPE_CHECKING:
    from aitestkit.agentic.memory import MemoryStore
    from aitestkit.judge.judge import Judge
    from aitestkit.providers.base import Embedder
    from aitestkit.redact import Redactor
    from aitestkit.vectordb.base import VectorStore


class MetricContext:
    """Shared services handed to metrics during evaluation."""

    def __init__(
        self,
        *,
        judge: Judge | None = None,
        embedder: Embedder | None = None,
        embed_model: str | None = None,
        redactor: Redactor | None = None,
        memory_store: MemoryStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.judge = judge
        self.embedder = embedder
        self.embed_model = embed_model
        self.redactor = redactor
        self.memory_store = memory_store
        self.vector_store = vector_store

    def derive(self, **overrides: Any) -> MetricContext:
        """Copy with field overrides — per-case stores/registries."""
        return MetricContext(
            judge=overrides.get("judge", self.judge),
            embedder=overrides.get("embedder", self.embedder),
            embed_model=overrides.get("embed_model", self.embed_model),
            redactor=overrides.get("redactor", self.redactor),
            memory_store=overrides.get("memory_store", self.memory_store),
            vector_store=overrides.get("vector_store", self.vector_store),
        )

    def require_judge(self, metric_name: str) -> Judge:
        if self.judge is None:
            raise ValueError(f"metric {metric_name!r} requires a judge; none configured")
        return self.judge

    def require_embedder(self, metric_name: str) -> Embedder:
        if self.embedder is None:
            raise ValueError(f"metric {metric_name!r} requires an embedder; none configured")
        return self.embedder


class BaseMetric(ABC):
    """One evaluation check. Subclasses implement ``measure``."""

    name: str = "base"
    required_fields: tuple[str, ...] = ()
    default_threshold: float = 0.5
    requires_judge: bool = False

    def __init__(self, *, threshold: float | None = None) -> None:
        self.threshold = self.default_threshold if threshold is None else threshold
        # Set by build_metric from the suite spec — see MetricSpec.
        self.negated = False
        self.informational = False
        self.required = False

    def missing_fields(self, case: TestCase) -> list[str]:
        """Fields this metric needs that the case doesn't provide."""
        missing = []
        for field in self.required_fields:
            value = getattr(case, field, None)
            if value is None:
                missing.append(field)
        return missing

    async def evaluate(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        """Public entry: handles skip-for-missing-fields then delegates."""
        missing = self.missing_fields(case)
        if missing:
            result = MetricResult.skipped(
                metric=self.name,
                reason=f"case missing required fields: {', '.join(missing)}",
            )
            if self.required:
                return MetricResult.error(
                    metric=self.name,
                    reason=f"required metric skipped: {result.reason}",
                )
            return result
        result = await self.measure(case, output, ctx)
        if self.required and result.status == MetricStatus.SKIPPED:
            return MetricResult.error(
                metric=self.name,
                reason=f"required metric skipped: {result.reason}",
            )
        if self.informational:
            result.details["informational"] = True
        if (
            self.negated
            and result.status == MetricStatus.SCORED
            and result.passed is not None
        ):
            result.passed = not result.passed
            result.details["negated"] = True
            result.reason = f"(negated) {result.reason}".strip()
        return result

    @abstractmethod
    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        """Compute the metric. Implementations return a scored MetricResult."""
        ...
