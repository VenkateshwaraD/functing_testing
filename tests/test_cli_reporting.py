"""CLI e2e + reporter coverage: run a suite, verify all report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitestkit.models import (
    CaseResult,
    CaseStatus,
    MetricResult,
    MetricStatus,
    TestRun,
)
from aitestkit.redact import Redactor
from aitestkit.reporting import attribution, console, coverage, github_actions


def _suite_files(tmp_path: Path, *, min_pass_rate: float = 0.9) -> Path:
    (tmp_path / "datasets").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "cases.yaml").write_text(
        """cases:
  - id: ok
    input: "what is my checking balance"
    expected_output: "your checking balance is $1250.00"
    tags: [banking]
  - id: bad
    input: "what is the daily ach limit"
    expected_output: "one million dollars"
    tags: [banking]
"""
    )
    suite = tmp_path / f"suite-{min_pass_rate}.yaml"
    suite.write_text(
        f"""name: cli-e2e
target:
  type: python
  function: demo_app:answer
datasets:
  - datasets/cases.yaml
metrics:
  - exact_match
judge:
  provider: fake
  model: fake
gates:
  min_pass_rate: {min_pass_rate}
"""
    )
    return suite


def test_cli_run_produces_all_artifacts(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from aitestkit.cli import app

    suite = _suite_files(tmp_path)
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["run", str(suite), "--out", str(out_dir), "--report", "json,junit,html"]
    )
    # 1/2 pass (0.5) < min_pass_rate 0.9 → gate fails → exit 5 (GATE_FAILURE)
    assert result.exit_code == 5, result.output
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "junit.xml").exists()
    assert (out_dir / "aitestkit-report.html").exists()
    payload = json.loads((out_dir / "results.json").read_text())
    assert payload["schema_version"] == 1
    statuses = {r["case_id"]: r["status"] for r in payload["results"]}
    assert statuses["ok"] == "passed"
    assert statuses["bad"] == "failed"
    assert "PASS ok" in result.output
    assert "FAIL bad" in result.output


def test_cli_gate_tolerated_failures_exit_ok(tmp_path: Path) -> None:
    """A failed case inside the declared tolerance exits 0 — gates are
    authoritative, not raw case outcomes."""
    from typer.testing import CliRunner

    from aitestkit.cli import app

    suite = _suite_files(tmp_path, min_pass_rate=0.4)
    result = CliRunner().invoke(app, ["run", str(suite), "--out", str(tmp_path / "out")])
    # 0.5 >= 0.4 → the gate tolerates the 'bad' failure
    assert result.exit_code == 0, result.output
    assert "FAIL bad" in result.output


def test_cli_config_error_exit(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from aitestkit.cli import app

    bad_suite = tmp_path / "bad.yaml"
    bad_suite.write_text("name: x\n")
    result = CliRunner().invoke(app, ["run", str(bad_suite)])
    assert result.exit_code == 2  # ExitCode.CONFIG_ERROR


def test_console_report_lines() -> None:
    result = CaseResult(
        case_id="c1",
        status=CaseStatus.FAILED,
        input="q",
        metrics=[
            MetricResult(
                metric="exact_match",
                status=MetricStatus.SCORED,
                score=0.0,
                passed=False,
                reason="mismatch",
            )
        ],
    )
    line = console._line_for(result)
    assert "FAIL c1" in line
    assert "exact_match=0.00" in line


def test_attribution_retrieval_vs_generation() -> None:
    retrieval_bad = CaseResult(
        case_id="r",
        status=CaseStatus.FAILED,
        input="q",
        metrics=[
            MetricResult(
                metric="recall_at_k", status=MetricStatus.SCORED, score=0.2, passed=False
            ),
            MetricResult(
                metric="faithfulness", status=MetricStatus.SCORED, score=0.9, passed=True
            ),
        ],
    )
    assert attribution.attribute_failure(retrieval_bad) == "retrieval"
    gen_bad = CaseResult(
        case_id="g",
        status=CaseStatus.FAILED,
        input="q",
        metrics=[
            MetricResult(
                metric="recall_at_k", status=MetricStatus.SCORED, score=0.9, passed=True
            ),
            MetricResult(
                metric="faithfulness",
                status=MetricStatus.SCORED,
                score=0.2,
                passed=False,
            ),
        ],
    )
    assert attribution.attribute_failure(gen_bad) == "generation"
    empty = CaseResult(case_id="e", status=CaseStatus.FAILED, input="q")
    assert attribution.attribute_failure(empty) is None


def test_coverage_matrix_includes_gaps() -> None:
    test_run = TestRun(
        run_id="r",
        suite_name="s",
        results=[
            CaseResult(
                case_id="c1",
                status=CaseStatus.PASSED,
                input="q",
                tags=["tools"],
            )
        ],
    )
    matrix = coverage.coverage_matrix(
        test_run, coverage_map={"tools": "Tool use", "memory": "Memory"}
    )
    assert matrix["Tool use"]["passed"] == 1
    assert matrix["Tool use"]["pass_rate"] == 1.0
    assert matrix["Memory"]["total"] == 0  # visible gap


def test_github_actions_summary_and_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    test_run = TestRun(
        run_id="r1",
        suite_name="ci-suite",
        results=[
            CaseResult(
                case_id="broken",
                status=CaseStatus.FAILED,
                input="q",
                metrics=[
                    MetricResult(
                        metric="exact_match",
                        status=MetricStatus.SCORED,
                        score=0.0,
                        passed=False,
                        reason="mismatch",
                    )
                ],
            )
        ],
    )
    github_actions.emit(test_run, redactor=Redactor(), suite_path=Path("suite.yaml"))
    text = summary.read_text()
    assert "ci-suite" in text
    out = capsys.readouterr().out
    assert "::error" in out
    assert "broken" in out


def test_github_actions_noop_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    test_run = TestRun(run_id="r", suite_name="s")
    github_actions.emit(test_run, redactor=Redactor())  # must not raise


def test_junit_renders_error_and_flaky(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    from aitestkit.reporting.junit import write_junit

    run = TestRun(
        run_id="r",
        suite_name="s",
        results=[
            CaseResult(
                case_id="err",
                status=CaseStatus.ERROR,
                input="q",
                error="timeout",
            ),
            CaseResult(
                case_id="flk",
                status=CaseStatus.FLAKY,
                input="q",
                repeat={"runs": 3, "pass_rate": 0.33},
            ),
        ],
    )
    path = tmp_path / "junit.xml"
    write_junit(run, path, redactor=Redactor())
    root = ET.parse(path).getroot()  # noqa: S314
    assert root.get("errors") == "1"
    assert root.find(".//error").get("message") == "timeout"
    flaky_el = root.find(".//flakyFailure")
    assert flaky_el is not None
    assert "flaky" in flaky_el.get("message")
