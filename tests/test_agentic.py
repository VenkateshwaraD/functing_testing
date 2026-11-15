"""Agentic pillar tests: emitter, trajectory diff, agentic metrics."""

from __future__ import annotations

import pytest

from aitestkit.metrics.agentic import (
    ConfirmationRequired,
    ForbiddenTools,
    HandoffCorrectness,
    LoopDetection,
    ReasoningCoherence,
    TaskCompletion,
    TerminationCheck,
    ToolArgsCorrect,
    ToolPreconditions,
    ToolSequence,
    TrajectoryMatch,
)
from aitestkit.metrics.base import MetricContext
from aitestkit.models import (
    MetricStatus,
    TargetOutput,
    TestCase,
    ToolCall,
    Trajectory,
)
from aitestkit.tracing.emitter import TraceEmitter
from aitestkit.tracing.recorder import diff_trajectories


def _trajectory(tools: list[str] | None = None, **kw) -> Trajectory:
    emitter = TraceEmitter()
    for name in tools or []:
        emitter.tool_call(name, args=kw.get("args", {}).get(name, {}))
    if kw.get("finish"):
        emitter.finish(kw["finish"])
    return emitter.trajectory


def _case(**kw) -> TestCase:
    return TestCase(id="a1", input="book a flight", **kw)


async def test_tool_sequence_subsequence() -> None:
    case = _case(expected_tools=["search", "book"])
    output = TargetOutput(trajectory=_trajectory(["search", "book"]))
    result = await ToolSequence().measure(case, output, MetricContext())
    assert result.passed is True
    # order violation: 'book' ran before 'search' so it doesn't count — 1/2 in order
    output2 = TargetOutput(trajectory=_trajectory(["book", "search"]))
    result2 = await ToolSequence().measure(case, output2, MetricContext())
    assert result2.score == 0.5
    diff = result2.details["tool_call_diff"]
    assert "book" in diff["missing"]


async def test_tool_sequence_partial_and_extra() -> None:
    case = _case(expected_tools=["a", "b", "c"])
    output = TargetOutput(trajectory=_trajectory(["a", "x", "b"]))
    result = await ToolSequence().measure(case, output, MetricContext())
    assert result.score == pytest.approx(2 / 3)
    assert result.details["tool_call_diff"]["unexpected"] == ["x"]


async def test_tool_args_correct() -> None:
    case = _case(
        expected_tools=[ToolCall(name="book", args={"dst": "paris"})]
    )
    ok = TargetOutput(
        tool_calls=[ToolCall(name="book", args={"dst": "paris", "seat": "1A"})]
    )
    result = await ToolArgsCorrect().measure(case, ok, MetricContext())
    assert result.passed is True
    bad = TargetOutput(tool_calls=[ToolCall(name="book", args={"dst": "london"})])
    result2 = await ToolArgsCorrect().measure(case, bad, MetricContext())
    assert result2.passed is False
    mismatches = result2.details["tool_call_diff"]["mismatches"]
    assert mismatches[0]["expected"] == "paris"


async def test_tool_preconditions() -> None:
    metric = ToolPreconditions(requires={"refund": ["verify_identity"]})
    good = TargetOutput(trajectory=_trajectory(["verify_identity", "refund"]))
    assert (await metric.measure(_case(), good, MetricContext())).passed is True
    bad = TargetOutput(trajectory=_trajectory(["refund", "verify_identity"]))
    result = await metric.measure(_case(), bad, MetricContext())
    assert result.passed is False
    assert result.details["violations"][0]["missing_prereq"] == "verify_identity"


async def test_forbidden_tools() -> None:
    metric = ForbiddenTools(tools=["drop_table", "rm_rf"])
    clean = TargetOutput(trajectory=_trajectory(["search", "read"]))
    assert (await metric.measure(_case(), clean, MetricContext())).passed is True
    dirty = TargetOutput(trajectory=_trajectory(["search", "drop_table"]))
    result = await metric.measure(_case(), dirty, MetricContext())
    assert result.passed is False
    assert result.details["forbidden_used"] == ["drop_table"]


async def test_confirmation_required() -> None:
    metric = ConfirmationRequired(tools=["delete"])
    # flagged via ToolCall.requires_confirmation — passes
    flagged = TargetOutput(
        tool_calls=[ToolCall(name="delete", requires_confirmation=True)]
    )
    assert (await metric.measure(_case(), flagged, MetricContext())).passed is True
    # confirmation event before call — passes
    emitter = TraceEmitter()
    emitter.emit("confirmation", prompt="delete all?")
    emitter.tool_call("delete")
    confirmed = TargetOutput(trajectory=emitter.trajectory)
    assert (await metric.measure(_case(), confirmed, MetricContext())).passed is True
    # bare call — fails
    bare = TargetOutput(tool_calls=[ToolCall(name="delete")])
    result = await metric.measure(_case(), bare, MetricContext())
    assert result.passed is False


async def test_loop_detection() -> None:
    emitter = TraceEmitter()
    for _ in range(4):
        emitter.tool_call("search", args={"q": "same"})
    output = TargetOutput(trajectory=emitter.trajectory)
    result = await LoopDetection().measure(_case(), output, MetricContext())
    assert result.passed is False
    assert result.details["loop"]["signature"].endswith(":search")
    ok = TargetOutput(trajectory=_trajectory(["a", "b", "c"]))
    assert (await LoopDetection().measure(_case(), ok, MetricContext())).passed is True


async def test_termination_check() -> None:
    done = TargetOutput(trajectory=_trajectory(["a"], finish="done"))
    assert (await TerminationCheck().measure(_case(), done, MetricContext())).passed is True
    capped = TargetOutput(trajectory=_trajectory(["a"], finish="max_steps"))
    result = await TerminationCheck().measure(_case(), capped, MetricContext())
    assert result.passed is False
    no_traj = TargetOutput()
    skipped = await TerminationCheck().measure(_case(), no_traj, MetricContext())
    assert skipped.status == MetricStatus.SKIPPED


async def test_trajectory_match() -> None:
    golden = _trajectory(["search", "select", "book"])
    case = _case(expected_trajectory=golden)
    identical = TargetOutput(trajectory=_trajectory(["search", "select", "book"]))
    result = await TrajectoryMatch().measure(case, identical, MetricContext())
    assert result.score == 1.0
    diverged = TargetOutput(trajectory=_trajectory(["search", "cancel"]))
    result2 = await TrajectoryMatch().measure(case, diverged, MetricContext())
    assert result2.details["trajectory_diff"]["first_divergence"] == 1
    assert result2.score < 1.0


async def test_handoff_correctness() -> None:
    emitter = TraceEmitter()
    emitter.handoff("planner", from_agent="supervisor")
    emitter.handoff("executor", from_agent="planner")
    case = _case(metadata={"expected_handoffs": ["planner", "executor"]})
    output = TargetOutput(trajectory=emitter.trajectory)
    result = await HandoffCorrectness().measure(case, output, MetricContext())
    assert result.passed is True
    missing = _case(metadata={"expected_handoffs": ["planner", "auditor"]})
    result2 = await HandoffCorrectness().measure(missing, output, MetricContext())
    assert result2.score == pytest.approx(0.5)


async def test_judge_agentic_metrics() -> None:
    from aitestkit.judge.judge import Judge
    from aitestkit.providers.fake import FakeChatProvider

    provider = FakeChatProvider()
    judge = Judge(provider, model="fake")
    ctx = MetricContext(judge=judge)
    emitter = TraceEmitter()
    emitter.thought("I should search for flights")
    emitter.tool_call("search")
    output = TargetOutput(output="booked", trajectory=emitter.trajectory)
    task = await TaskCompletion().measure(_case(), output, ctx)
    assert task.status == MetricStatus.SCORED
    reasoning = await ReasoningCoherence().measure(_case(), output, ctx)
    assert reasoning.status == MetricStatus.SCORED


async def test_reasoning_skipped_without_thoughts() -> None:
    from aitestkit.judge.judge import Judge
    from aitestkit.providers.fake import FakeChatProvider

    ctx = MetricContext(judge=Judge(FakeChatProvider(), model="fake"))
    output = TargetOutput(output="done", trajectory=_trajectory(["a"]))
    result = await ReasoningCoherence().measure(_case(), output, ctx)
    assert result.status == MetricStatus.SKIPPED
    assert "hidden CoT" in result.reason


def test_diff_trajectories() -> None:
    expected = _trajectory(["a", "b", "c"])
    actual = _trajectory(["a", "x"])
    diff = diff_trajectories(expected, actual)
    assert diff["first_divergence"] == 1
    assert diff["identical"] is False
    same = diff_trajectories(expected, _trajectory(["a", "b", "c"]))
    assert same["identical"] is True
