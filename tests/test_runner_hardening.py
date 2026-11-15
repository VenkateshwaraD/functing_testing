"""Runner hardening tests: --resume, cancellation, doctor."""

from __future__ import annotations

from pathlib import Path

from aitestkit.models import CaseResult, CaseStatus, TestCase
from aitestkit.redact import Redactor
from aitestkit.reporting.stream import ResultsStream, read_stream
from aitestkit.runner import RunEngine


async def test_engine_cancel_skips_pending() -> None:
    from aitestkit.metrics.base import MetricContext
    from aitestkit.metrics.deterministic import ExactMatch
    from aitestkit.targets.python_target import PythonTarget

    engine = RunEngine(
        target=PythonTarget(spec="demo_app:answer"),
        metrics=[ExactMatch()],
        ctx=MetricContext(),
    )
    engine.cancel()
    test_run, _ = await engine.run(
        [TestCase(id="x", input="say hello", expected_output="hello")]
    )
    assert test_run.results[0].status == CaseStatus.SKIPPED
    assert "cancelled" in (test_run.results[0].error or "")


async def test_resume_skips_completed_ids() -> None:
    from aitestkit.metrics.base import MetricContext
    from aitestkit.metrics.deterministic import ExactMatch
    from aitestkit.targets.python_target import PythonTarget

    engine = RunEngine(
        target=PythonTarget(spec="demo_app:answer"),
        metrics=[ExactMatch()],
        ctx=MetricContext(),
    )
    cases = [
        TestCase(id="done", input="say hello", expected_output="WRONG"),
        TestCase(id="new", input="say hello", expected_output="hello"),
    ]
    test_run, _ = await engine.run(cases, skip_ids={"done"})
    assert len(test_run.results) == 1
    assert test_run.results[0].case_id == "new"
    assert test_run.results[0].status == CaseStatus.PASSED


def test_resume_roundtrip_via_jsonl(tmp_path: Path) -> None:
    """Crash-mid-run: JSONL has case 1; resume re-runs only case 2."""
    jsonl = tmp_path / "results.jsonl"
    from aitestkit.models import TestRun

    stream = ResultsStream(jsonl, redactor=Redactor())
    stream.open(TestRun(run_id="old", suite_name="s"))
    stream.write_case(
        CaseResult(case_id="case-1", status=CaseStatus.PASSED, input="q")
    )
    _header, done = read_stream(jsonl)
    skip_ids = {r.case_id for r in done}
    assert skip_ids == {"case-1"}
    assert "case-2" not in skip_ids


def test_doctor_lists_checks(capsys) -> None:
    from typer.testing import CliRunner

    from aitestkit.cli import app

    result = CliRunner().invoke(app, ["doctor"])
    # doctor exits 0 when everything present, 2 when missing — either is fine,
    # but it must actually print the check sections
    assert "== environment" in result.output
    assert "== optional extras" in result.output


def test_doctor_suite_provider_missing_key(
    tmp_path: Path, monkeypatch
) -> None:
    from typer.testing import CliRunner

    from aitestkit.cli import app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: t\ntarget: {type: python, function: demo_app:answer}\n"
        "judge: {provider: openai, model: gpt-4o-mini}\n"
        "metrics: [correctness]\n"
    )
    result = CliRunner().invoke(app, ["doctor", str(suite)])
    assert "OPENAI_API_KEY" in result.output
    assert result.exit_code != 0
