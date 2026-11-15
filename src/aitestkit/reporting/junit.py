"""JUnit XML writer — feeds CI test-report integrations (GitLab included)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from aitestkit.models import CaseResult, CaseStatus, TestRun
from aitestkit.redact import Redactor


def write_junit(test_run: TestRun, path: Path, *, redactor: Redactor) -> None:
    """Write results as junit.xml: failures+errors marked, others pass/skip."""
    suite = ET.Element(
        "testsuite",
        {
            "name": test_run.suite_name or "aitestkit",
            "tests": str(len(test_run.results)),
            "failures": str(sum(r.status == CaseStatus.FAILED for r in test_run.results)),
            "errors": str(sum(r.status == CaseStatus.ERROR for r in test_run.results)),
            "skipped": str(sum(r.status == CaseStatus.SKIPPED for r in test_run.results)),
            "time": f"{sum(r.latency_ms for r in test_run.results) / 1000:.3f}",
        },
    )
    for result in test_run.results:
        case_el = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": test_run.suite_name or "aitestkit",
                "name": result.case_id,
                "time": f"{result.latency_ms / 1000:.3f}",
            },
        )
        _write_case_body(case_el, result, redactor)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _write_case_body(
    case_el: ET.Element, result: CaseResult, redactor: Redactor
) -> None:
    failed = [m for m in result.metrics if m.passed is False]
    if result.status == CaseStatus.FAILED:
        reasons = "; ".join(f"{m.metric}: {m.reason}" for m in failed)
        failure = ET.SubElement(case_el, "failure", {"message": reasons or "failed"})
        failure.text = redactor.redact_text(reasons)
    elif result.status == CaseStatus.ERROR:
        error = ET.SubElement(
            case_el, "error", {"message": result.error or "infrastructure error"}
        )
        error.text = redactor.redact_text(result.error or "")
    elif result.status in (CaseStatus.SKIPPED,):
        ET.SubElement(case_el, "skipped", {"message": result.error or "skipped"})
    elif result.status == CaseStatus.FLAKY:
        flaky = ET.SubElement(
            case_el,
            "flakyFailure",
            {"message": f"flaky: {result.repeat.get('pass_rate', 0):.0%} pass rate"},
        )
        flaky.text = str(result.repeat.get("statuses", []))
