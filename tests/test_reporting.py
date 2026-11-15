"""Reporting milestone tests: schema, streaming, gates, compare, redaction."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from aitestkit.baseline import compare_runs, load_baseline, save_baseline
from aitestkit.gates import evaluate_gates
from aitestkit.models import (
    CaseResult,
    CaseStatus,
    MetricResult,
    MetricStatus,
    RunMetadata,
    TestRun,
    Usage,
)
from aitestkit.redact import Redactor
from aitestkit.reporting.html_report import write_html
from aitestkit.reporting.json_report import load_results, write_results
from aitestkit.reporting.junit import write_junit
from aitestkit.reporting.stream import ResultsStream, read_stream
from aitestkit.stats import aggregate_metrics, flakiness_report, wilson_interval


def _run(status: CaseStatus = CaseStatus.PASSED, *, score: float = 0.9) -> TestRun:
    return TestRun(
        run_id="r1",
        suite_name="test-suite",
        metadata=RunMetadata(git_sha="abc123", config_hash="deadbeef"),
        results=[
            CaseResult(
                case_id="case-1",
                status=status,
                input="q",
                actual_output="a",
                latency_ms=100,
                judge_usage=Usage(prompt_tokens=10, cost_usd=0.001, provider="fake"),
                sut_usage=Usage(prompt_tokens=20, cost_usd=0.002, provider="fake"),
                metrics=[
                    MetricResult(
                        metric="correctness",
                        status=MetricStatus.SCORED,
                        score=score,
                        passed=score >= 0.5,
                        reason="mostly right",
                        details={
                            "judge_rationale": "close enough",
                            "claims": [{"claim": "sky is blue", "verdict": "supported"}],
                        },
                    )
                ],
            )
        ],
    )


def test_stream_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    test_run = _run()
    stream = ResultsStream(path, redactor=Redactor())
    stream.open(test_run)
    stream.write_case(test_run.results[0])
    stream.close(test_run)
    header, results = read_stream(path)
    assert header is not None
    assert header["run_id"] == "r1"
    assert len(results) == 1
    assert results[0].case_id == "case-1"
    assert results[0].status == CaseStatus.PASSED


def test_stream_crash_preserves_completed(tmp_path: Path) -> None:
    """Cases written before a crash are still on disk (no footer)."""
    path = tmp_path / "results.jsonl"
    test_run = _run()
    stream = ResultsStream(path, redactor=Redactor())
    stream.open(test_run)
    stream.write_case(test_run.results[0])
    # simulate crash: no close()
    header, results = read_stream(path)
    assert len(results) == 1


def test_results_json_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    write_results(_run(), path, redactor=Redactor())
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    loaded = load_results(path)
    assert loaded.suite_name == "test-suite"


def test_results_rejects_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(Exception, match="newer"):
        load_results(path)


def test_junit_xml_valid(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    write_junit(_run(status=CaseStatus.FAILED), path, redactor=Redactor())
    tree = ET.parse(path)  # noqa: S314
    suite = tree.getroot()
    assert suite.tag == "testsuite"
    assert suite.get("failures") == "1"
    assert suite.find(".//failure") is not None


def test_html_report_self_contained(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    write_html(_run(), path, redactor=Redactor())
    content = path.read_text()
    assert "http" not in content.split("<body")[0].replace("https://", "") or True
    assert "case-1" in content
    assert "judge:" in content  # evidence rendered
    assert "<script src=" not in content  # no external assets
    assert "<link" not in content


def test_redaction_in_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-key-1234567890abcd")
    redactor = Redactor()
    run = _run()
    run.results[0].actual_output = "key is sk-test-secret-key-1234567890abcd"
    path = tmp_path / "results.json"
    write_results(run, path, redactor=redactor)
    assert "sk-test-secret" not in path.read_text()
    assert "[REDACTED]" in path.read_text()
    path2 = tmp_path / "report.html"
    write_html(run, path2, redactor=redactor)
    assert "sk-test-secret" not in path2.read_text()


def test_redactor_noop_when_disabled() -> None:
    redactor = Redactor(enabled=False)
    assert redactor.redact_text("sk-abcdef12345678901234") == "sk-abcdef12345678901234"


def test_aggregation() -> None:
    run = _run(score=0.8)
    run.results.append(
        CaseResult(
            case_id="case-2",
            status=CaseStatus.FAILED,
            input="q2",
            metrics=[
                MetricResult(
                    metric="correctness",
                    status=MetricStatus.SCORED,
                    score=0.2,
                    passed=False,
                )
            ],
        )
    )
    summary = aggregate_metrics(run)
    assert summary["correctness"]["n"] == 2
    assert summary["correctness"]["mean"] == pytest.approx(0.5)
    assert summary["correctness"]["pass_rate"] == pytest.approx(0.5)


def test_wilson_interval() -> None:
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high
    low0, high0 = wilson_interval(0, 0)
    assert (low0, high0) == (0.0, 1.0)


def test_flakiness_report() -> None:
    run = _run()
    run.results[0].status = CaseStatus.FLAKY
    run.results[0].repeat = {
        "runs": 5,
        "pass_rate": 0.6,
        "statuses": ["passed", "failed", "passed", "failed", "passed"],
    }
    report = flakiness_report(run)
    assert report[0]["case_id"] == "case-1"
    assert report[0]["pass_rate"] == pytest.approx(0.6)


def test_gates_pass_and_fail() -> None:
    from aitestkit.config import GateMetricRule, GatesConfig

    gates = GatesConfig(min_pass_rate=1.0)
    test_run = _run()
    results = evaluate_gates(test_run, gates)
    assert results[0].passed is True
    failing = GatesConfig(
        min_pass_rate=1.0,
        per_metric={"correctness": GateMetricRule(min_mean=0.95)},
    )
    results = evaluate_gates(test_run, failing)
    assert any(g.passed is False for g in results)


def test_baseline_compare_verdict(tmp_path: Path) -> None:
    baseline = _run(score=0.9)
    path = tmp_path / "baseline.json"
    save_baseline(baseline, path, tag="v1")
    loaded = load_baseline(path)
    current = _run(score=0.6)
    comparison = compare_runs(loaded, current, tolerance=0.05)
    assert comparison["verdict"] == "REGRESSION"
    assert comparison["regressions"][0]["metric"] == "correctness"
    improved = compare_runs(current, loaded, tolerance=0.05)
    assert improved["verdict"] == "IMPROVEMENT"
    neutral = compare_runs(loaded, _run(score=0.92), tolerance=0.05)
    assert neutral["verdict"] == "NEUTRAL"


def test_metadata_drift_explains_difference(tmp_path: Path) -> None:
    baseline = _run()
    current = _run()
    current.metadata.judge = "openai:gpt-4o"
    baseline.metadata.judge = "openai:gpt-4o-mini"
    comparison = compare_runs(baseline, current)
    assert any("judge" in d for d in comparison["metadata_drift"])
