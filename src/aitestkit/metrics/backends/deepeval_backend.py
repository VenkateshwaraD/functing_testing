"""DeepEval backend — wraps deepeval metrics behind BaseMetric.

Requires ``aitestkit[deepeval]``. Maps our TestCase/TargetOutput to
deepeval's LLMTestCase; score/pass semantics are preserved so gates and
reports work identically to native metrics.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase

# metric spec name -> deepeval class name
_DEEPEVAL_METRICS = {
    "faithfulness": "FaithfulnessMetric",
    "answer_relevancy": "AnswerRelevancyMetric",
    "contextual_recall": "ContextualRecallMetric",
    "contextual_precision": "ContextualPrecisionMetric",
    "contextual_relevancy": "ContextualRelevancyMetric",
    "hallucination": "HallucinationMetric",
    "bias": "BiasMetric",
    "toxicity": "ToxicityMetric",
    "summarization": "SummarizationMetric",
}


@register
class DeepEvalMetric(BaseMetric):
    """Run any deepeval metric. Params: ``metric`` name + ``threshold``.

    Example suite entry::

        - name: deepeval
          params: {metric: faithfulness, threshold: 0.85}
    """

    name = "deepeval"
    default_threshold = 0.5

    def __init__(
        self,
        *,
        metric: str = "faithfulness",
        threshold: float | None = None,
        **metric_kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold)
        self.deepeval_metric = metric
        self.metric_kwargs = metric_kwargs

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        if importlib.util.find_spec("deepeval") is None:
            return MetricResult.error(
                metric=self.name,
                reason="deepeval backend requires: pip install 'aitestkit[deepeval]'",
            )
        class_name = _DEEPEVAL_METRICS.get(self.deepeval_metric)
        if class_name is None:
            return MetricResult.error(
                metric=self.name,
                reason=(
                    f"unknown deepeval metric {self.deepeval_metric!r}; "
                    f"available: {sorted(_DEEPEVAL_METRICS)}"
                ),
            )
        try:
            import deepeval.metrics as de_metrics
            from deepeval.test_case import LLMTestCase
        except ImportError as e:
            return MetricResult.error(metric=self.name, reason=str(e))
        metric_cls = getattr(de_metrics, class_name, None)
        if metric_cls is None:
            return MetricResult.error(
                metric=self.name,
                reason=f"deepeval has no {class_name} in this version",
            )
        context = output.retrieval_context or case.context or []
        test_case = LLMTestCase(
            input=case.input,
            actual_output=output.output or "",
            expected_output=case.expected_output,
            retrieval_context=context or None,
            context=context or None,
        )
        try:
            instance = metric_cls(
                threshold=self.threshold, **self.metric_kwargs
            )
            instance.measure(test_case)
        except Exception as e:  # noqa: BLE001 — provider errors stay errors
            return MetricResult.error(
                metric=self.name, reason=f"deepeval measure failed: {e}"
            )
        return MetricResult(
            metric=f"deepeval_{self.deepeval_metric}",
            status=MetricStatus.SCORED,
            score=float(instance.score),
            passed=bool(instance.success),
            reason=str(getattr(instance, "reason", "") or ""),
            details={
                "backend": "deepeval",
                "metric": self.deepeval_metric,
                "verbose_logs": getattr(instance, "verbose_logs", None),
            },
        )
