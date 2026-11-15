"""Statistics: metric aggregation, tag pass rates, confidence intervals, flakiness.

Regression comparisons require a delta beyond ``tolerance`` AND non-overlapping
confidence intervals (or a significant two-proportion test) — judge noise must
not produce false regressions.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from aitestkit.models import CaseResult, CaseStatus, MetricStatus, TestRun


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sample."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


def wilson_interval(successes: int, trials: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)
    z = 1.96 if alpha >= 0.05 else 2.58
    p_hat = successes / trials
    denom = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def aggregate_metrics(test_run: TestRun) -> dict[str, dict[str, Any]]:
    """Per-metric aggregation: mean/median/p10/p90, scored/error/skipped counts."""
    buckets: dict[str, list[float]] = {}
    statuses: dict[str, dict[str, int]] = {}
    pass_counts: dict[str, int] = {}
    for result in test_run.results:
        for metric in result.metrics:
            status_counts = statuses.setdefault(
                metric.metric, {"scored": 0, "error": 0, "skipped": 0}
            )
            status_counts[metric.status.value] = status_counts.get(metric.status.value, 0) + 1
            if metric.status != MetricStatus.SCORED or metric.score is None:
                continue
            buckets.setdefault(metric.metric, []).append(metric.score)
            if metric.passed:
                pass_counts[metric.metric] = pass_counts.get(metric.metric, 0) + 1
    summary: dict[str, dict[str, Any]] = {}
    for name, scores in buckets.items():
        summary[name] = {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
            "p10": percentile(scores, 10),
            "p90": percentile(scores, 90),
            "min": min(scores),
            "max": max(scores),
            "pass_rate": pass_counts.get(name, 0) / len(scores),
            "statuses": statuses[name],
        }
    return summary


def pass_rates_by_tag(test_run: TestRun) -> dict[str, dict[str, Any]]:
    """Pass/fail/error counts grouped by case tag."""
    groups: dict[str, dict[str, Any]] = {}
    for result in test_run.results:
        for tag in result.tags or ["untagged"]:
            counts = groups.setdefault(tag, {"total": 0})
            counts["total"] += 1
            counts[result.status.value] = counts.get(result.status.value, 0) + 1
    for _tag, counts in groups.items():
        total = counts["total"]
        counts["pass_rate"] = counts.get(CaseStatus.PASSED.value, 0) / total if total else 0.0
    return groups


def flakiness_report(test_run: TestRun) -> list[dict[str, Any]]:
    """Rank cases by instability: lowest pass_rate among repeated runs first."""
    unstable = []
    for result in test_run.results:
        runs = result.repeat.get("runs", 1)
        if runs < 2:
            continue
        pass_rate = result.repeat.get("pass_rate", 0.0)
        unstable.append(
            {
                "case_id": result.case_id,
                "runs": runs,
                "pass_rate": pass_rate,
                "status": result.status.value,
                "statuses": result.repeat.get("statuses", []),
            }
        )
    return sorted(unstable, key=lambda item: item["pass_rate"])


def case_pass_interval(result: CaseResult, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson CI for a case's repeat pass rate."""
    runs = result.repeat.get("runs", 1)
    passes = round(result.repeat.get("pass_rate", 0.0) * runs)
    return wilson_interval(passes, runs, alpha=alpha)


def is_significant_regression(
    *,
    baseline_rate: tuple[int, int],
    current_rate: tuple[int, int],
    tolerance: float,
    alpha: float = 0.05,
) -> bool:
    """True if current is meaningfully below baseline beyond tolerance + noise.

    ``*_rate`` args are (successes, trials). The regression must exceed the
    tolerance AND the current CI's upper bound must sit below the baseline
    point estimate minus tolerance.
    """
    b_successes, b_trials = baseline_rate
    c_successes, c_trials = current_rate
    if b_trials == 0:
        return False
    baseline_p = b_successes / b_trials
    current_p = c_successes / c_trials if c_trials else 0.0
    if baseline_p - current_p <= tolerance:
        return False
    _, current_upper = wilson_interval(c_successes, c_trials, alpha=alpha)
    return current_upper < baseline_p - tolerance
