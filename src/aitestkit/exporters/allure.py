"""Allure exporter — writes Allure results JSON per case.

No SDK needed: the allure-results format is plain JSON files in a
directory that ``allure generate`` consumes. Steps map to metric
evaluations so failures show the metric name + reason in the report.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from aitestkit.models import CaseResult, TestRun


class AllureExporter:
    """One allure-results dir per run — attach to CI artifacts."""

    name = "allure"

    def export(self, test_run: TestRun, out_dir: Path) -> str | None:
        results_dir = out_dir / "allure-results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for result in test_run.results:
            payload = _case_payload(result)
            (results_dir / f"{uuid.uuid4()}-result.json").write_text(
                json.dumps(payload, indent=2)
            )
        (results_dir / "environment.properties").write_text(
            f"suite={test_run.suite_name}\nrun_id={test_run.run_id}\n"
            f"git_sha={test_run.metadata.git_sha or 'unknown'}\n"
        )
        return f"wrote {len(test_run.results)} allure results to {results_dir}"


def _status_name(result: CaseResult) -> str:
    mapping = {
        "passed": "passed",
        "failed": "failed",
        "flaky": "failed",
        "error": "broken",
        "skipped": "skipped",
    }
    return mapping.get(result.status.value, "broken")


def _case_payload(result: CaseResult) -> dict:
    now_ms = int(time.time() * 1000)
    steps = [
        {
            "name": f"{metric.metric}: {metric.status.value}"
            + (f" score={metric.score:.2f}" if metric.score is not None else ""),
            "status": (
                "passed" if metric.passed
                else "failed" if metric.passed is False
                else "skipped"
            ),
            "statusDetails": {"message": metric.reason} if metric.reason else {},
            "stage": "finished",
        }
        for metric in result.metrics
    ]
    return {
        "uuid": str(uuid.uuid4()),
        "name": result.case_id,
        "status": _status_name(result),
        "statusDetails": {"message": result.error} if result.error else {},
        "stage": "finished",
        "steps": steps,
        "start": now_ms - int(result.latency_ms),
        "stop": now_ms,
        "labels": [
            {"name": "suite", "value": "aitestkit"},
            *({"name": "tag", "value": t} for t in result.tags),
        ],
    }
