"""Console reporter: per-case status lines plus a summary table.

This is the CLI's presentation boundary — it writes with typer.echo, which
is the only place user-facing output lives (library code uses logging).
"""

from __future__ import annotations

import typer

from aitestkit.models import CaseResult, CaseStatus, TestRun

_STATUS_MARK = {
    CaseStatus.PASSED: "PASS",
    CaseStatus.FAILED: "FAIL",
    CaseStatus.ERROR: "ERR ",
    CaseStatus.SKIPPED: "SKIP",
    CaseStatus.FLAKY: "FLKY",
    CaseStatus.XFAIL: "XFLK",
}


def _line_for(result: CaseResult) -> str:
    mark = _STATUS_MARK.get(result.status, "????")
    failed_metrics = [
        f"{m.metric}={m.score:.2f}" if m.score is not None else f"{m.metric}=err"
        for m in result.metrics
        if m.passed is False
    ]
    detail = f"  ({', '.join(failed_metrics)})" if failed_metrics else ""
    error = f"  [{result.error}]" if result.error else ""
    return f"{mark} {result.case_id}{detail}{error}"


def report_case(result: CaseResult) -> None:
    """Echo one case line as it completes."""
    typer.echo(_line_for(result))


def report_summary(test_run: TestRun) -> None:
    """Echo the end-of-run summary: counts, latency, cost."""
    counts: dict[str, int] = {}
    total_latency = 0.0
    judge_cost = 0.0
    sut_cost = 0.0
    for result in test_run.results:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
        total_latency += result.latency_ms
        judge_cost += result.judge_usage.cost_usd
        sut_cost += result.sut_usage.cost_usd
    total = len(test_run.results)
    parts = [f"{status}={count}" for status, count in sorted(counts.items())]
    typer.echo("")
    typer.echo(f"== {test_run.suite_name or 'run'}: {total} cases — {' '.join(parts)}")
    typer.echo(
        f"   errors={counts.get('error', 0)} flaky={counts.get('flaky', 0)}"
    )
    typer.echo(
        f"   latency {total_latency:.0f}ms total | "
        f"sut ${sut_cost:.4f} | judge ${judge_cost:.4f}"
    )
    for gate in test_run.gates:
        mark = "PASS" if gate.passed else "FAIL"
        typer.echo(f"   gate {mark} {gate.name}: {gate.actual} ({gate.expected})")
