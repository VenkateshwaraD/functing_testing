"""Requirement-coverage view: capability x status matrix.

Cases carry capability tags (tools, memory, handoffs, chunking, retrieval,
vectordb, multiturn, data-quality...). The suite's ``coverage:`` section maps
tags to display names; capabilities with zero cases show as gaps.
"""

from __future__ import annotations

from typing import Any

from aitestkit.models import CaseStatus, TestRun


def coverage_matrix(
    test_run: TestRun, *, coverage_map: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build capability -> {cases, status counts, pass_rate} mapping.

    ``coverage_map`` maps case tags to capability display names; tags not in
    the map are shown under their raw tag name. All declared capabilities
    appear even with zero cases (visible gaps).
    """
    coverage_map = coverage_map or {}
    matrix: dict[str, dict[str, Any]] = {
        name: {"total": 0} for name in coverage_map.values()
    }
    for result in test_run.results:
        capabilities = [coverage_map.get(t, t) for t in result.tags] or ["untagged"]
        for capability in capabilities:
            bucket = matrix.setdefault(capability, {"total": 0})
            bucket["total"] += 1
            key = result.status.value
            bucket[key] = bucket.get(key, 0) + 1
    for bucket in matrix.values():
        total = bucket["total"]
        bucket["pass_rate"] = (
            bucket.get(CaseStatus.PASSED.value, 0) / total if total else 0.0
        )
    return matrix
