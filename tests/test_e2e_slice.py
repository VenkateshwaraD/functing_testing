"""Vertical-slice e2e: python target -> fake judge -> metrics -> runner -> report."""

from __future__ import annotations

import pytest

from aitestkit.judge.judge import Judge
from aitestkit.metrics.base import MetricContext
from aitestkit.metrics.deterministic import ExactMatch
from aitestkit.metrics.llm_judge import Correctness
from aitestkit.models import CaseStatus, TestCase
from aitestkit.providers.fake import FakeChatProvider
from aitestkit.runner import RunEngine
from aitestkit.targets.python_target import PythonTarget


@pytest.fixture
def engine() -> RunEngine:
    provider = FakeChatProvider()
    provider.respond_to("capital of france", {"score": 0.95, "reason": "correct answer"})
    provider.respond_to("2+2", {"score": 0.3, "reason": "wrong answer"})
    judge = Judge(provider, model="fake-judge")
    ctx = MetricContext(judge=judge)
    return RunEngine(
        target=PythonTarget(spec="demo_app:answer"),
        metrics=[ExactMatch(), Correctness(threshold=0.7)],
        ctx=ctx,
    )


async def test_vertical_slice_e2e(engine: RunEngine) -> None:
    """The full path works: target -> metrics (det + judge) -> statuses."""
    cases = [
        TestCase(id="ok", input="what is my checking balance", expected_output="your checking balance is $1250.00"),
        TestCase(id="bad", input="what is 2+2", expected_output="five"),
    ]
    test_run, budget_hit = await engine.run(cases)
    assert not budget_hit
    assert len(test_run.results) == 2
    by_id = {r.case_id: r for r in test_run.results}
    assert by_id["ok"].status == CaseStatus.PASSED
    # 'bad': exact_match fails (4 != five), correctness judge gives 0.3 < 0.7
    assert by_id["bad"].status == CaseStatus.FAILED
    correctness = next(m for m in by_id["bad"].metrics if m.metric == "correctness")
    assert correctness.score == pytest.approx(0.3)
    assert correctness.passed is False


async def test_error_status_on_broken_target() -> None:
    """A crashing target produces ERROR, not FAILED."""
    from aitestkit.exceptions import ConfigError

    with pytest.raises(ConfigError):
        PythonTarget(spec="demo_app:does_not_exist")
    engine = RunEngine(
        target=_BrokenTarget(),
        metrics=[ExactMatch()],
        ctx=MetricContext(),
    )
    test_run, _ = await engine.run([TestCase(id="x", input="q", expected_output="a")])
    assert test_run.results[0].status == CaseStatus.ERROR


class _BrokenTarget:
    name = "broken"

    async def invoke(self, case, **kwargs):
        from aitestkit.exceptions import TargetError

        raise TargetError("boom")


async def test_skipped_when_metric_fields_missing(engine: RunEngine) -> None:
    """A case without expected_output -> correctness skipped, exact too."""
    cases = [TestCase(id="noexp", input="what is the daily ach limit")]
    test_run, _ = await engine.run(cases)
    statuses = {m.metric: m.status.value for m in test_run.results[0].metrics}
    assert statuses["exact_match"] == "skipped"
    assert statuses["correctness"] == "skipped"
    assert test_run.results[0].status == CaseStatus.SKIPPED
