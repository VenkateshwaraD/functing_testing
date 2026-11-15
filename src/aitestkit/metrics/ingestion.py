"""Ingestion stage checks — verify the corpus pipeline before retrieval.

These run on documents the case supplies in ``case.metadata['documents']``
(list of ``{id, text, metadata}``) or on ``case.context`` (plain texts).
They catch pipeline bugs — unparsed binary junk, missing metadata,
duplicates — before retrieval metrics ever see bad chunks.
"""

from __future__ import annotations

import hashlib
from typing import Any

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.registry import register
from aitestkit.models import MetricResult, MetricStatus, TargetOutput, TestCase


def _documents(case: TestCase) -> list[dict[str, Any]]:
    docs = case.metadata.get("documents")
    if isinstance(docs, list):
        return [d for d in docs if isinstance(d, dict)]
    return [{"id": str(i), "text": t} for i, t in enumerate(case.context or [])]


def _texts(case: TestCase) -> list[str]:
    docs = _documents(case)
    return [str(d.get("text", "")) for d in docs]


@register
class ParseCheck(BaseMetric):
    """Documents must be non-empty, decodable text — no binary garbage."""

    name = "ingestion_parse"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        texts = _texts(case)
        if not texts:
            return MetricResult.skipped(
                metric=self.name, reason="no documents/context supplied"
            )
        bad = [
            i for i, t in enumerate(texts)
            if not t.strip() or "\ufffd" in t or "\x00" in t
        ]
        score = 1.0 - len(bad) / len(texts)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={
                "total": len(texts),
                "unparseable": bad,
                "attribution_hint": "ingestion",
            },
        )


@register
class MetadataCompleteness(BaseMetric):
    """Required metadata keys present on every document.

    Case metadata: ``documents`` list of dicts, plus ``required_metadata``
    list of key names each doc must carry.
    """

    name = "ingestion_metadata"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        required = case.metadata.get("required_metadata") or []
        docs = _documents(case)
        if not docs:
            return MetricResult.skipped(
                metric=self.name, reason="no documents supplied"
            )
        missing: list[dict[str, object]] = []
        for doc in docs:
            meta = doc.get("metadata") or {}
            absent = [k for k in required if k not in meta]
            if absent:
                missing.append({"id": doc.get("id"), "missing": absent})
        score = 1.0 - len(missing) / len(docs)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={
                "required": required,
                "docs_missing_metadata": missing,
                "attribution_hint": "ingestion",
            },
        )


@register
class DeduplicationCheck(BaseMetric):
    """Duplicate content detection — exact hash + normalized-text hashing."""

    name = "ingestion_dedup"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        texts = _texts(case)
        if not texts:
            return MetricResult.skipped(
                metric=self.name, reason="no documents/context supplied"
            )
        seen: dict[str, int] = {}
        dupes: list[int] = []
        for i, text in enumerate(texts):
            digest = hashlib.sha256(
                " ".join(text.lower().split()).encode()
            ).hexdigest()
            if digest in seen:
                dupes.append(i)
            else:
                seen[digest] = i
        score = 1.0 - len(dupes) / len(texts)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=score,
            passed=score >= self.threshold,
            details={
                "total": len(texts),
                "duplicate_indexes": dupes,
                "attribution_hint": "ingestion",
            },
        )
