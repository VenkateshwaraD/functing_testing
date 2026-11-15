"""Automation tests: prompt matrix, prompt_sim metric, prompt snapshots."""

from __future__ import annotations

from pathlib import Path

from aitestkit.config import PromptVariant
from aitestkit.metrics.base import MetricContext
from aitestkit.metrics.deterministic import Contains
from aitestkit.metrics.prompts import PromptSim
from aitestkit.models import CaseStatus, TargetOutput, TestCase
from aitestkit.prompts import (
    diff_prompt_snapshots,
    load_prompt_snapshot,
    render_prompt,
    save_prompt_snapshot,
    snapshot_prompts,
    unresolved_placeholders,
)
from aitestkit.runner import RunEngine
from aitestkit.targets.python_target import PythonTarget


def test_render_and_unresolved() -> None:
    case = TestCase(id="t", input="hi", metadata={"who": "devin"})
    assert render_prompt("Say {input} to {who}", case) == "Say hi to devin"
    rendered = render_prompt("Say {input} to {missing}", case)
    assert unresolved_placeholders(rendered) == ["missing"]


async def test_prompt_sim_metric() -> None:
    metric = PromptSim(template="Answer {input} for {user}")
    case = TestCase(id="t", input="hi", metadata={"user": "ada"})
    ok = await metric.measure(case, TargetOutput(), MetricContext())
    assert ok.passed is True
    assert ok.details["rendered_prompt"] == "Answer hi for ada"

    bad = await PromptSim(
        template="Answer {input} {missing}"
    ).measure(case, TargetOutput(), MetricContext())
    assert bad.passed is False
    assert "unresolved" in bad.reason

    banned = await PromptSim(
        template="you are a helpful assistant {input}",
        banned=["helpful"],
    ).measure(case, TargetOutput(), MetricContext())
    assert banned.passed is False
    over = await PromptSim(template="{input}", max_tokens=1).measure(
        TestCase(id="t", input="many " * 50), TargetOutput(), MetricContext()
    )
    assert over.passed is False


async def test_prompt_matrix_expansion() -> None:
    """Each case runs once per prompt variant; target sees rendered prompt."""
    import demo_app

    demo_app.SEEN_PROMPTS.clear()
    engine = RunEngine(
        target=PythonTarget(spec="demo_app:prompt_echo"),
        metrics=[Contains(substrings=["echo"])],
        ctx=MetricContext(),
        prompts=[
            PromptVariant(name="a", template="T1: {input}"),
            PromptVariant(name="b", template="T2: {input}"),
        ],
    )
    test_run, _ = await engine.run([TestCase(id="c1", input="x")])
    assert len(test_run.results) == 2
    ids = sorted(r.case_id for r in test_run.results)
    assert ids == ["c1[a]", "c1[b]"]
    assert sorted(demo_app.SEEN_PROMPTS) == [("a", "T1: x"), ("b", "T2: x")]
    assert all(r.status == CaseStatus.PASSED for r in test_run.results)
    assert test_run.results[0].metadata["prompt_name"] in {"a", "b"}


def test_prompt_snapshot_and_diff(tmp_path: Path) -> None:
    cases = [TestCase(id="c1", input="x"), TestCase(id="c2", input="y")]
    v1 = [PromptVariant(name="p", template="A: {input}")]
    snap1 = snapshot_prompts(cases, v1)
    path = tmp_path / "prompts.json"
    save_prompt_snapshot(snap1, path)
    loaded = load_prompt_snapshot(path)
    assert loaded == snap1

    v2 = [PromptVariant(name="p", template="B: {input}")]
    snap2 = snapshot_prompts(cases, v2)
    diff = diff_prompt_snapshots(snap1, snap2)
    assert diff["changed_count"] == 2
    assert sorted(diff["changed_prompts"]) == ["c1", "c2"]

    same = diff_prompt_snapshots(snap1, snap1)
    assert same["changed_count"] == 0
