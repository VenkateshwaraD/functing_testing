"""RAG failure attribution: which pipeline stage likely caused a bad answer.

Rule-based verdict from the metric pattern — the payoff of stage-level
testing. Low retrieval + faithful-but-wrong -> retrieval; good retrieval +
low faithfulness -> generation; retrieved but deprioritized -> rerank.
"""

from __future__ import annotations

from aitestkit.models import CaseResult, MetricStatus, TestRun

_RETRIEVAL_METRICS = {"recall_at_k", "precision_at_k", "mrr", "ndcg", "hit_rate"}
_RERANK_METRICS = {"rerank_delta", "rerank_ndcg"}
_GENERATION_METRICS = {"faithfulness", "groundedness", "answer_relevancy", "citation_accuracy"}
_INGESTION_METRICS = {"chunk_quality", "ingestion_parse", "dedup"}
_CONTEXT_METRICS = {"context_window", "contextual_precision", "contextual_recall"}


def attribute_failure(result: CaseResult) -> str | None:
    """Return the likely failing stage for a failed RAG case, else None."""
    scored = {
        m.metric: m.score
        for m in result.metrics
        if m.status == MetricStatus.SCORED and m.score is not None
    }
    if not scored:
        return None
    retrieval_bad = _any_below(scored, _RETRIEVAL_METRICS, 0.5)
    rerank_bad = _any_below(scored, _RERANK_METRICS, 0.0)
    generation_bad = _any_below(scored, _GENERATION_METRICS, 0.5)
    ingestion_bad = _any_below(scored, _INGESTION_METRICS, 0.5)
    context_bad = _any_below(scored, _CONTEXT_METRICS, 0.5)
    if ingestion_bad:
        return "ingestion/chunking"
    if retrieval_bad:
        return "retrieval"
    if rerank_bad:
        return "rerank"
    if context_bad and not retrieval_bad:
        return "context_window"
    if generation_bad:
        return "generation"
    return None


def _any_below(scores: dict[str, float], names: set[str], floor: float) -> bool:
    return any(score < floor for name, score in scores.items() if name in names)


def apply_attribution(test_run: TestRun) -> int:
    """Set ``attribution`` on every failed case; returns how many were set."""
    attributed = 0
    for result in test_run.results:
        if result.attribution:
            attributed += 1
            continue
        label = attribute_failure(result)
        if label:
            result.attribution = label
            attributed += 1
    return attributed
