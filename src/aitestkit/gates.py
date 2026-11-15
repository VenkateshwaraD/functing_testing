"""Quality gates: suite-level release criteria evaluated after a run.

A failed gate drives exit code 5 (GATE_FAILURE). Rules cover overall pass
rate, per-metric aggregates (mean/p10/p95), per-case score floors, error and
flaky budgets, regression-vs-baseline tolerance, and coverage minimums.
"""

from __future__ import annotations

from typing import Any

from aitestkit.config import GateMetricRule, GatesConfig
from aitestkit.models import CaseStatus, GateResult, MetricStatus, TestRun
from aitestkit.stats import aggregate_metrics


def evaluate_gates(
    test_run: TestRun,
    gates: GatesConfig,
    *,
    regression_actual: float | None = None,
) -> list[GateResult]:
    """Evaluate all configured gates against a finished run."""
    results: list[GateResult] = []
    results.extend(_pass_rate_gate(test_run, gates))
    results.extend(_error_gate(test_run, gates))
    results.extend(_flaky_gate(test_run, gates))
    results.extend(_metric_gates(test_run, gates))
    results.extend(_coverage_gates(test_run, gates))
    results.extend(_regression_gate(gates, regression_actual))
    return results


def _pass_rate_gate(test_run: TestRun, gates: GatesConfig) -> list[GateResult]:
    if gates.min_pass_rate is None:
        return []
    judged = [r for r in test_run.results if r.status in {CaseStatus.PASSED, CaseStatus.FAILED}]
    rate = (
        sum(r.status == CaseStatus.PASSED for r in judged) / len(judged)
        if judged
        else 0.0
    )
    return [
        GateResult(
            name="min_pass_rate",
            passed=rate >= gates.min_pass_rate,
            actual=round(rate, 4),
            expected=f">= {gates.min_pass_rate}",
            detail=f"{len(judged)} judged cases",
        )
    ]


def _error_gate(test_run: TestRun, gates: GatesConfig) -> list[GateResult]:
    if gates.max_error_cases is None:
        return []
    errors = sum(r.status == CaseStatus.ERROR for r in test_run.results)
    return [
        GateResult(
            name="max_error_cases",
            passed=errors <= gates.max_error_cases,
            actual=errors,
            expected=f"<= {gates.max_error_cases}",
        )
    ]


def _flaky_gate(test_run: TestRun, gates: GatesConfig) -> list[GateResult]:
    if gates.max_flaky is None:
        return []
    total = len(test_run.results)
    flaky = sum(r.status == CaseStatus.FLAKY for r in test_run.results)
    fraction = flaky / total if total else 0.0
    return [
        GateResult(
            name="max_flaky",
            passed=fraction <= gates.max_flaky,
            actual=round(fraction, 4),
            expected=f"<= {gates.max_flaky}",
        )
    ]


def _metric_gates(test_run: TestRun, gates: GatesConfig) -> list[GateResult]:
    summary = aggregate_metrics(test_run)
    results: list[GateResult] = []
    for metric_name, rule in gates.per_metric.items():
        stats = summary.get(metric_name)
        if stats is None:
            results.append(
                GateResult(
                    name=f"metric:{metric_name}",
                    passed=False,
                    expected="present in results",
                    detail="metric produced no scored results",
                )
            )
            continue
        results.extend(_metric_rule(metric_name, rule, stats, test_run))
    return results


def _metric_rule(
    metric_name: str,
    rule: GateMetricRule,
    stats: dict[str, Any],
    test_run: TestRun,
) -> list[GateResult]:
    results: list[GateResult] = []
    checks = [
        ("min_mean", stats["mean"], lambda v, t: v >= t, ">="),
        ("min_p10", stats["p10"], lambda v, t: v >= t, ">="),
        ("max_p95", None, lambda v, t: v <= t, "<="),
    ]
    for attr, value, op, symbol in checks:
        bound = getattr(rule, attr, None)
        if bound is None:
            continue
        if attr == "max_p95":
            scores = [
                m.score
                for r in test_run.results
                for m in r.metrics
                if m.metric == metric_name
                and m.status == MetricStatus.SCORED
                and m.score is not None
            ]
            from aitestkit.stats import percentile

            value = percentile(scores, 95)
        results.append(
            GateResult(
                name=f"{metric_name}.{attr}",
                passed=op(value, bound),
                actual=round(value, 4),
                expected=f"{symbol} {bound}",
            )
        )
    if rule.min_per_case is not None:
        worst = stats["min"]
        results.append(
            GateResult(
                name=f"{metric_name}.min_per_case",
                passed=worst >= rule.min_per_case,
                actual=round(worst, 4),
                expected=f">= {rule.min_per_case}",
                detail="lowest individual case score",
            )
        )
    return results


def _coverage_gates(test_run: TestRun, gates: GatesConfig) -> list[GateResult]:
    results: list[GateResult] = []
    for capability, minimum in gates.min_coverage.items():
        covered = sum(1 for r in test_run.results if capability in r.tags)
        results.append(
            GateResult(
                name=f"coverage:{capability}",
                passed=covered >= minimum,
                actual=covered,
                expected=f">= {minimum} cases",
            )
        )
    return results


def _regression_gate(
    gates: GatesConfig, regression_actual: float | None
) -> list[GateResult]:
    if gates.max_regression_vs_baseline is None:
        return []
    if regression_actual is None:
        return [
            GateResult(
                name="max_regression_vs_baseline",
                passed=False,
                expected=f"<= {gates.max_regression_vs_baseline}",
                detail="no baseline comparison was performed",
            )
        ]
    return [
        GateResult(
            name="max_regression_vs_baseline",
            passed=regression_actual <= gates.max_regression_vs_baseline,
            actual=round(regression_actual, 4),
            expected=f"<= {gates.max_regression_vs_baseline}",
        )
    ]


def all_gates_passed(gates: list[GateResult]) -> bool:
    """True when every evaluated gate passed (vacuous if none configured)."""
    return all(g.passed for g in gates)
