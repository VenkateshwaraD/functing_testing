"""Exporter tests: allure output format, otel lazy error, factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitestkit.exceptions import ConfigError
from aitestkit.exporters.allure import AllureExporter
from aitestkit.exporters.factory import build_exporters
from aitestkit.models import (
    CaseResult,
    CaseStatus,
    MetricResult,
    TestRun,
)


def _run() -> TestRun:
    return TestRun(
        run_id="r1",
        suite_name="demo",
        results=[
            CaseResult(
                case_id="c1",
                status=CaseStatus.PASSED,
                input="hi",
                actual_output="hello",
                latency_ms=12.5,
                metrics=[
                    MetricResult(
                        metric="contains", score=1.0, passed=True, reason=""
                    )
                ],
            ),
            CaseResult(
                case_id="c2",
                status=CaseStatus.FAILED,
                input="bye",
                actual_output="wrong",
                metrics=[
                    MetricResult(
                        metric="exact_match",
                        score=0.0,
                        passed=False,
                        reason="expected 'goodbye'",
                    )
                ],
            ),
        ],
    )


def test_allure_writes_results(tmp_path: Path) -> None:
    note = AllureExporter().export(_run(), tmp_path)
    assert note and "2 allure results" in note
    results_dir = tmp_path / "allure-results"
    files = list(results_dir.glob("*-result.json"))
    assert len(files) == 2
    payload = json.loads(files[0].read_text())
    assert payload["status"] in {"passed", "failed"}
    assert payload["steps"], "metric evaluations become allure steps"
    env = (results_dir / "environment.properties").read_text()
    assert "suite=demo" in env
    assert "run_id=r1" in env


def test_otel_requires_extra() -> None:
    try:
        import opentelemetry  # noqa: F401

        pytest.skip("otel installed")
    except ImportError:
        pass
    from aitestkit.exporters.otel import OtelExporter

    with pytest.raises(ConfigError, match="aitestkit\\[otel\\]"):
        OtelExporter()


def test_factory_unknown_exporter() -> None:
    with pytest.raises(ConfigError, match="unknown exporter"):
        build_exporters(["bogus"])


def test_factory_allure() -> None:
    exporters = build_exporters(["allure"])
    assert exporters[0].name == "allure"
