"""RAGAS backend — wraps ragas metrics behind BaseMetric.

Requires ``aitestkit[ragas]``. Single-sample evaluation: our case maps to a
ragas SingleTurnSample; score lands on the standard MetricResult.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase

# metric spec name -> ragas class
_RAGAS_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "ResponseRelevancy",
    "context_recall": "LLMContextRecall",
    "context_precision": "LLMContextPrecisionWithoutReference",
    "context_utilization": "ContextUtilization",
    "answer_correctness": "AnswerCorrectness",
    "answer_similarity": "SemanticSimilarity",
}


@register
class RagasMetric(BaseMetric):
    """Run any ragas metric. Params: ``metric`` name + ``threshold``.

    Example suite entry::

        - name: ragas
          params: {metric: faithfulness, threshold: 0.85}
    """

    name = "ragas"
    default_threshold = 0.5

    def __init__(
        self,
        *,
        metric: str = "faithfulness",
        threshold: float | None = None,
        **metric_kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold)
        self.ragas_metric = metric
        self.metric_kwargs = metric_kwargs

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        if importlib.util.find_spec("ragas") is None:
            return MetricResult.error(
                metric=self.name,
                reason="ragas backend requires: pip install 'aitestkit[ragas]'",
            )
        class_name = _RAGAS_METRICS.get(self.ragas_metric)
        if class_name is None:
            return MetricResult.error(
                metric=self.name,
                reason=(
                    f"unknown ragas metric {self.ragas_metric!r}; "
                    f"available: {sorted(_RAGAS_METRICS)}"
                ),
            )
        try:
            import ragas.metrics as ragas_metrics
            from ragas.dataset_schema import SingleTurnSample
        except ImportError:
            try:
                from ragas import SingleTurnSample
            except ImportError as e:
                return MetricResult.error(metric=self.name, reason=str(e))
            import ragas.metrics as ragas_metrics
        metric_cls = getattr(ragas_metrics, class_name, None)
        if metric_cls is None:
            return MetricResult.error(
                metric=self.name,
                reason=f"ragas has no {class_name} in this version",
            )
        sample = SingleTurnSample(
            user_input=case.input,
            response=output.output or "",
            reference=case.expected_output,
            retrieved_contexts=output.retrieval_context or case.context or [],
        )
        try:
            instance = metric_cls(**self.metric_kwargs)
            score = await instance.single_turn_ascore(sample)
        except Exception as e:  # noqa: BLE001 — provider errors stay errors
            return MetricResult.error(
                metric=self.name, reason=f"ragas score failed: {e}"
            )
        score_f = float(score)
        return MetricResult(
            metric=f"ragas_{self.ragas_metric}",
            status=MetricStatus.SCORED,
            score=score_f,
            passed=score_f >= self.threshold,
            details={
                "backend": "ragas",
                "metric": self.ragas_metric,
            },
        )
