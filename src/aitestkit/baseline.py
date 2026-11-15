"""Baselines: versioned reference runs for regression comparison.

A baseline is a saved TestRun tagged by model/provider/prompt/kb/embedder/
chunker versions — embedder and chunker swaps are exactly the regressions
worth catching. ``compare`` produces a verdict, not a raw diff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aitestkit.models import CaseResult, CaseStatus, MetricStatus, TestRun
from aitestkit.stats import aggregate_metrics

DEFAULT_TOLERANCE = 0.05


def save_baseline(test_run: TestRun, path: Path, *, tag: str) -> None:
    """Persist a run as a named baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    test_run.baseline_tag = tag
    path.write_text(test_run.model_dump_json(indent=2))


def load_baseline(path: Path) -> TestRun:
    """Load a baseline run (same schema as results.json)."""
    if not path.exists():
        raise FileNotFoundError(f"baseline not found: {path}")
    return TestRun.model_validate(json.loads(path.read_text()))


def _metric_scores(result: CaseResult) -> dict[str, float]:
    return {
        m.metric: m.score
        for m in result.metrics
        if m.status == MetricStatus.SCORED and m.score is not None
    }


def _status_rank(status: CaseStatus) -> int:
    order = {
        CaseStatus.PASSED: 3,
        CaseStatus.FLAKY: 2,
        CaseStatus.FAILED: 1,
        CaseStatus.SKIPPED: 0,
        CaseStatus.ERROR: 0,
    }
    return order.get(status, 0)


def compare_runs(
    baseline: TestRun,
    current: TestRun,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare current against baseline: regressions, improvements, verdict.

    A metric drop counts as a regression only when it exceeds ``tolerance``;
    under repeat-N, CI significance also applies (handled upstream by stats).
    """
    baseline_cases = {r.case_id: r for r in baseline.results}
    current_cases = {r.case_id: r for r in current.results}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    for case_id, current_result in sorted(current_cases.items()):
        base_result = baseline_cases.get(case_id)
        if base_result is None:
            matrix.append({"case_id": case_id, "status": "new"})
            continue
        row = _compare_case(base_result, current_result, tolerance=tolerance)
        matrix.append(row)
        regressions.extend(row["regressions"])
        improvements.extend(row["improvements"])
    verdict = _verdict(regressions, improvements)
    return {
        "verdict": verdict,
        "regressions": regressions,
        "improvements": improvements,
        "matrix": matrix,
        "missing_in_current": sorted(set(baseline_cases) - set(current_cases)),
        "metadata_drift": _metadata_drift(baseline, current),
        "metric_summary_delta": _metric_delta(baseline, current),
    }


def _compare_case(
    base: CaseResult, current: CaseResult, *, tolerance: float
) -> dict[str, Any]:
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    base_scores = _metric_scores(base)
    current_scores = _metric_scores(current)
    for metric, current_score in current_scores.items():
        base_score = base_scores.get(metric)
        if base_score is None:
            continue
        delta = current_score - base_score
        entry = {
            "case_id": current.case_id,
            "metric": metric,
            "baseline": round(base_score, 4),
            "current": round(current_score, 4),
            "delta": round(delta, 4),
        }
        if delta < -tolerance:
            regressions.append(entry)
        elif delta > tolerance:
            improvements.append(entry)
    if _status_rank(current.status) < _status_rank(base.status):
        regressions.append(
            {
                "case_id": current.case_id,
                "metric": "_status",
                "baseline": base.status.value,
                "current": current.status.value,
                "delta": "status regression",
            }
        )
    return {
        "case_id": current.case_id,
        "baseline_status": base.status.value,
        "current_status": current.status.value,
        "regressions": regressions,
        "improvements": improvements,
    }


def _verdict(regressions: list[dict[str, Any]], improvements: list[dict[str, Any]]) -> str:
    if regressions:
        return "REGRESSION"
    if improvements:
        return "IMPROVEMENT"
    return "NEUTRAL"


def _metadata_drift(baseline: TestRun, current: TestRun) -> list[str]:
    """Fields that differ between runs — explains why results differ."""
    drift: list[str] = []
    b_meta, c_meta = baseline.metadata, current.metadata
    for field in ("judge", "embedder", "aitestkit_version", "python_version", "seed"):
        if getattr(b_meta, field) != getattr(c_meta, field):
            drift.append(
                f"{field}: {getattr(b_meta, field)} -> {getattr(c_meta, field)}"
            )
    for key in set(b_meta.model_versions) | set(c_meta.model_versions):
        if b_meta.model_versions.get(key) != c_meta.model_versions.get(key):
            drift.append(
                f"version.{key}: {b_meta.model_versions.get(key)} -> "
                f"{c_meta.model_versions.get(key)}"
            )
    if b_meta.config_hash != c_meta.config_hash:
        drift.append("config hash differs")
    return drift


def _metric_delta(baseline: TestRun, current: TestRun) -> dict[str, dict[str, float]]:
    """Aggregate per-metric mean deltas across the two runs."""
    baseline_stats = aggregate_metrics(baseline)
    current_stats = aggregate_metrics(current)
    deltas: dict[str, dict[str, float]] = {}
    for metric, stats in current_stats.items():
        base = baseline_stats.get(metric)
        if base is None:
            continue
        deltas[metric] = {
            "baseline_mean": round(base["mean"], 4),
            "current_mean": round(stats["mean"], 4),
            "delta": round(stats["mean"] - base["mean"], 4),
            "baseline_pass_rate": round(base["pass_rate"], 4),
            "current_pass_rate": round(stats["pass_rate"], 4),
        }
    return deltas


def max_metric_regression(comparison: dict[str, Any]) -> float:
    """Largest metric-score drop seen — feeds the regression gate."""
    drops = [
        -entry["delta"]
        for entry in comparison["regressions"]
        if isinstance(entry.get("delta"), (int, float))
    ]
    return max(drops, default=0.0)
