"""GitHub Actions integration: step summary + workflow annotations.

Auto-detected via ``GITHUB_ACTIONS`` — writes a markdown run summary to
``GITHUB_STEP_SUMMARY`` and emits ``::error``/``::warning`` annotations for
failed cases so they surface inline on PRs. Artifact upload is configured
in the workflow YAML itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aitestkit.models import CaseResult, CaseStatus, TestRun
from aitestkit.redact import Redactor
from aitestkit.stats import aggregate_metrics


def in_github_actions() -> bool:
    """True when running inside GitHub Actions."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def emit(test_run: TestRun, *, redactor: Redactor, suite_path: Path | None = None) -> None:
    """Write the step summary and emit annotations (no-op outside Actions)."""
    if not in_github_actions():
        return
    _write_step_summary(test_run, redactor)
    _emit_annotations(test_run, redactor, suite_path)


def _write_step_summary(test_run: TestRun, redactor: Redactor) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    counts: dict[str, int] = {}
    for result in test_run.results:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
    gate_line = _gate_verdict_line(test_run)
    metric_rows = _metric_rows(test_run)
    failures = [
        r for r in test_run.results if r.status in {CaseStatus.FAILED, CaseStatus.ERROR}
    ][:10]
    failure_lines = "".join(
        f"| `{_md(redactor.redact_text(r.case_id))}` | {r.status.value} | "
        f"{_md(redactor.redact_text(r.error or _fail_reason(r)))} |\n"
        for r in failures
    )
    text = (
        f"## aitestkit — {_md(redactor.redact_text(test_run.suite_name))}\n\n"
        f"{' · '.join(f'**{s}**: {c}' for s, c in sorted(counts.items()))}\n\n"
        f"{gate_line}\n\n"
        f"| metric | mean | p10 | pass rate |\n|---|---|---|---|\n{metric_rows}\n"
    )
    if failure_lines:
        text += (
            f"\n<details><summary>Top failures ({len(failures)})</summary>\n\n"
            f"| case | status | reason |\n|---|---|---|\n{failure_lines}</details>\n"
        )
    Path(summary_path).write_text(text, encoding="utf-8")


def _gate_verdict_line(test_run: TestRun) -> str:
    if not test_run.gates:
        return "_no quality gates configured_"
    failed = [g for g in test_run.gates if not g.passed]
    if not failed:
        return "**PASS** — all quality gates passed"
    names = ", ".join(g.name for g in failed)
    return f"**FAIL** — gates failed: {names}"


def _metric_rows(test_run: TestRun) -> str:
    summary = aggregate_metrics(test_run)
    return "".join(
        f"| `{name}` | {s['mean']:.3f} | {s['p10']:.3f} | {s['pass_rate']:.0%} |\n"
        for name, s in sorted(summary.items())
    )


def _emit_annotations(
    test_run: TestRun, redactor: Redactor, suite_path: Path | None
) -> None:
    """Emit ::error/::warning workflow commands for failing cases.

    Every interpolated field is escaped — an unescaped newline or ``::`` in a
    case id would let a dataset line inject arbitrary workflow commands.
    """
    location = f"file={_escape_property(str(suite_path))}," if suite_path else ""
    for result in test_run.results:
        case_id = _escape_property(redactor.redact_text(result.case_id))
        if result.status == CaseStatus.FAILED:
            message = _escape_command(redactor.redact_text(_fail_reason(result)))
            sys.stdout.write(
                f"::error {location}title=aitestkit::{case_id}: {message}\n"
            )
        elif result.status == CaseStatus.ERROR:
            message = _escape_command(redactor.redact_text(result.error or "error"))
            sys.stdout.write(
                f"::error {location}title=aitestkit-infra::{case_id}: {message}\n"
            )
        elif result.status == CaseStatus.FLAKY:
            sys.stdout.write(
                f"::warning {location}title=aitestkit-flaky::{case_id}: "
                f"flaky ({result.repeat.get('pass_rate', 0):.0%} pass rate)\n"
            )


def _fail_reason(result: CaseResult) -> str:
    reasons = [f"{m.metric}: {m.reason}" for m in result.metrics if m.passed is False]
    return "; ".join(reasons) or result.error or "failed"


def _escape_command(text: str) -> str:
    """Escape %, CR, LF for workflow command data."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    """Escape a workflow-command property value (``file=``, ``title=``).

    Property fields additionally require ``:`` and ``,`` escaping so the
    value cannot end the annotation block early.
    """
    return _escape_command(text).replace(":", "%3A").replace(",", "%2C")


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
