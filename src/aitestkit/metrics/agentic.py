"""Agentic metrics: tool use, trajectories, termination, handoffs, reasoning.

Deterministic checks (sequence, args, preconditions, forbidden tools,
confirmation, loops, termination, trajectory diff) plus judge-backed checks
(task completion, reasoning coherence, plan validity). Reasoning metrics
score only *emitted* thoughts — hidden chain-of-thought is never assumed.
"""

from __future__ import annotations

from typing import Any

from aitestkit.metrics.base import BaseMetric, MetricContext
from aitestkit.metrics.deterministic import _result
from aitestkit.metrics.registry import register
from aitestkit.models import (
    MetricResult,
    MetricStatus,
    TargetOutput,
    TestCase,
    ToolCall,
    Trajectory,
)
from aitestkit.tracing.recorder import diff_trajectories


def _calls(output: TargetOutput) -> list[ToolCall]:
    """Tool calls from trajectory + flat list, deduplicated by order."""
    calls = list(output.tool_calls)
    if output.trajectory:
        calls.extend(output.trajectory.tool_calls)
    return calls


def _expected_calls(case: TestCase) -> list[ToolCall]:
    out: list[ToolCall] = []
    for item in case.expected_tools or []:
        out.append(item if isinstance(item, ToolCall) else ToolCall(name=str(item)))
    return out


def _names(calls: list[ToolCall]) -> list[str]:
    return [c.name for c in calls]


@register
class ToolSequence(BaseMetric):
    """Expected tools called in order (subsequence) or exactly.

    Params: ``mode: subsequence|exact`` (default subsequence).
    """

    name = "tool_sequence"
    required_fields = ("expected_tools",)
    default_threshold = 1.0

    def __init__(self, *, mode: str = "subsequence", threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.mode = mode

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        expected = _names(_expected_calls(case))
        actual = _names(_calls(output))
        missing, unexpected = _sequence_diff(expected, actual, self.mode)
        score = 1.0 if not missing and (self.mode != "exact" or not unexpected) else 0.0
        if self.mode == "subsequence":
            score = (len(expected) - len(missing)) / len(expected) if expected else 1.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"{len(expected)-len(missing)}/{len(expected)} expected tools in order",
            details={
                "tool_call_diff": {
                    "expected": expected,
                    "actual": actual,
                    "missing": missing,
                    "unexpected": unexpected,
                }
            },
        )


def _sequence_diff(
    expected: list[str], actual: list[str], mode: str
) -> tuple[list[str], list[str]]:
    """Missing expected names + unexpected actual names (order-aware)."""
    if mode == "exact":
        missing = [e for e in expected if e not in actual]
        unexpected = [a for a in actual if a not in expected]
        order_ok = expected == actual
        if not order_ok and not missing and not unexpected:
            missing = ["<order mismatch>"]
        return missing, unexpected
    # subsequence: walk actual once
    iter_actual = iter(actual)
    remaining = list(expected)
    for name in iter_actual:
        if remaining and name == remaining[0]:
            remaining.pop(0)
    expected_set = set(expected)
    unexpected = [a for a in actual if a not in expected_set]
    return remaining, unexpected


@register
class ToolArgsCorrect(BaseMetric):
    """Each expected tool call's args must match (expected ⊆ actual)."""

    name = "tool_args_correct"
    required_fields = ("expected_tools",)
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        expected = _expected_calls(case)
        actual = _calls(output)
        diffs: list[dict[str, Any]] = []
        for exp in expected:
            call = next((c for c in actual if c.name == exp.name), None)
            if call is None:
                diffs.append({"tool": exp.name, "issue": "not called"})
                continue
            if exp.args:
                for key, value in exp.args.items():
                    if call.args.get(key) != value:
                        diffs.append(
                            {
                                "tool": exp.name,
                                "arg": str(key),
                                "expected": value,
                                "actual": call.args.get(key),
                            }
                        )
        score = 1.0 if not diffs else 0.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"{len(diffs)} argument/call mismatches" if diffs else "args correct",
            details={"tool_call_diff": {"mismatches": diffs}},
        )


@register
class ToolPreconditions(BaseMetric):
    """Named tools must be preceded by required tools (at the right time).

    Params: ``requires: {tool: [prereq, ...]}``.
    """

    name = "tool_preconditions"
    default_threshold = 1.0

    def __init__(
        self, *, requires: dict[str, list[str]], threshold: float | None = None
    ) -> None:
        super().__init__(threshold=threshold)
        self.requires = requires

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        names = _names(_calls(output))
        violations = []
        for tool, prereqs in self.requires.items():
            if tool not in names:
                continue
            index = names.index(tool)
            before = set(names[:index])
            for prereq in prereqs:
                if prereq not in before:
                    violations.append(
                        {"tool": tool, "missing_prereq": prereq}
                    )
        score = 1.0 if not violations else 0.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"{len(violations)} precondition violations" if violations else "order ok",
            details={"violations": violations, "observed_order": names},
        )


@register
class ForbiddenTools(BaseMetric):
    """Listed tools must never be called. Params: ``tools: [...]``."""

    name = "forbidden_tools"
    default_threshold = 1.0

    def __init__(self, *, tools: list[str], threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.tools = tools

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        actual = _names(_calls(output))
        hits = [t for t in actual if t in self.tools]
        score = 1.0 if not hits else 0.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"forbidden tools called: {hits}" if hits else "no forbidden calls",
            details={"forbidden_used": hits},
        )


@register
class ConfirmationRequired(BaseMetric):
    """Destructive tools need confirmation before use.

    A call passes if its ToolCall.requires_confirmation flag is set (the SUT
    asked) or a ``confirmation``/``message(role=confirmation)`` event
    precedes it. Params: ``tools: [...]``.
    """

    name = "confirmation_required"
    default_threshold = 1.0

    def __init__(self, *, tools: list[str], threshold: float | None = None) -> None:
        super().__init__(threshold=threshold)
        self.tools = tools

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        calls = _calls(output)
        events = output.trajectory.events if output.trajectory else []
        violations = []
        for call in calls:
            if call.name not in self.tools:
                continue
            if call.requires_confirmation:
                continue
            if not _confirmed_before(events, call.name):
                violations.append(call.name)
        score = 1.0 if not violations else 0.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"unconfirmed destructive calls: {violations}" if violations else "confirmed",
            details={"unconfirmed": violations},
        )


def _confirmed_before(events: list, tool_name: str) -> bool:
    """A confirmation event appears before the tool's call event."""
    call_index = next(
        (i for i, e in enumerate(events)
         if e.type == "tool_call" and e.payload.get("name") == tool_name),
        None,
    )
    if call_index is None:
        return False
    for event in events[:call_index]:
        if event.type == "confirmation":
            return True
        if event.type == "message" and event.payload.get("role") == "confirmation":
            return True
    return False


@register
class LoopDetection(BaseMetric):
    """Detects repeated step signatures and runaway step counts.

    Params: ``max_steps`` (default 50), ``max_repeats`` (default 3 — the same
    step signature repeated this many times consecutively is a loop).
    """

    name = "loop_detection"
    default_threshold = 1.0

    def __init__(
        self,
        *,
        max_steps: int = 50,
        max_repeats: int = 3,
        threshold: float | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.max_steps = max_steps
        self.max_repeats = max_repeats

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        trajectory = output.trajectory
        if trajectory is None:
            return MetricResult.skipped(metric=self.name, reason="no trajectory")
        sigs = trajectory.step_signatures
        loop = _find_loop(sigs, self.max_repeats)
        over_cap = len(sigs) > self.max_steps
        score = 0.0 if loop or over_cap else 1.0
        reason = []
        if loop:
            reason.append(f"loop at step {loop['start']}: {loop['signature']}")
        if over_cap:
            reason.append(f"{len(sigs)} steps > cap {self.max_steps}")
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason="; ".join(reason) if reason else f"{len(sigs)} steps, no loop",
            details={"loop": loop, "steps": len(sigs), "max_steps": self.max_steps},
        )


def _find_loop(signatures: list[str], max_repeats: int) -> dict[str, Any] | None:
    run_length = 1
    for index in range(1, len(signatures)):
        if signatures[index] == signatures[index - 1]:
            run_length += 1
            if run_length >= max_repeats:
                return {
                    "start": index - run_length + 1,
                    "signature": signatures[index],
                    "repeats": run_length,
                }
        else:
            run_length = 1
    return None


@register
class TerminationCheck(BaseMetric):
    """Run must terminate cleanly: finish_reason done (or absent), not
    max_steps/timeout/cancelled."""

    name = "termination_check"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        trajectory = output.trajectory
        if trajectory is None:
            return MetricResult.skipped(metric=self.name, reason="no trajectory")
        reason = trajectory.finish_reason or "done"
        bad = {"max_steps", "timeout", "cancelled", "error"}
        score = 0.0 if reason in bad else 1.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"finish_reason={reason}",
            details={"finish_reason": reason, "steps": len(trajectory.events)},
        )


@register
class TrajectoryMatch(BaseMetric):
    """Golden trajectory diff: expected vs actual step signatures.

    Score = fraction of expected steps matched before the first divergence.
    """

    name = "trajectory_match"
    required_fields = ("expected_trajectory",)
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        trajectory = output.trajectory
        if trajectory is None:
            return _result(
                self.name, score=0.0, threshold=self.threshold,
                reason="SUT produced no trajectory",
            )
        diff = diff_trajectories(case.expected_trajectory, trajectory)  # type: ignore[arg-type]
        total = diff["expected_steps"] or 1
        matched = diff["first_divergence"] if diff["first_divergence"] is not None else total
        score = matched / total
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=(
                f"diverged at step {diff['first_divergence']}"
                if diff["first_divergence"] is not None
                else "trajectory identical"
            ),
            details={"trajectory_diff": diff},
        )


@register
class HandoffCorrectness(BaseMetric):
    """Observed agent handoff sequence matches expected delegation order.

    Expected sequence comes from ``case.metadata['expected_handoffs']``.
    """

    name = "handoff_correctness"
    default_threshold = 1.0

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        expected = case.metadata.get("expected_handoffs")
        if not expected:
            return MetricResult.skipped(
                metric=self.name, reason="no expected_handoffs in case metadata"
            )
        actual = output.trajectory.agent_sequence if output.trajectory else []
        missing = [a for a in expected if a not in actual]
        score = (len(expected) - len(missing)) / len(expected) if expected else 1.0
        return _result(
            self.name,
            score=score,
            threshold=self.threshold,
            reason=f"handoffs {actual} vs expected {expected}",
            details={"expected": expected, "actual": actual, "missing": missing},
        )


_TASK_PROMPT = """You are evaluating an autonomous agent. Given the TASK and the
agent's TRAJECTORY + FINAL OUTPUT, score task completion from 0.0 (task not
accomplished) to 1.0 (fully accomplished). Judge the outcome, not effort.

TASK:
{input}

TRAJECTORY:
{trajectory}

FINAL OUTPUT:
{actual}

Respond with JSON only: {{"score": <float>, "reason": "<one sentence>",
"details": {{"accomplished": ["..."], "missing": ["..."]}}}}
"""

_REASONING_PROMPT = """You are evaluating an agent's EMITTED reasoning (the
thoughts it chose to record — never assume hidden reasoning). Score the
coherence of this chain toward solving the TASK from 0.0 (incoherent,
contradictory, or missing) to 1.0 (clear ReAct-style reasoning that
connects observations to actions).

TASK:
{input}

REASONING CHAIN:
{chain}

FINAL OUTPUT:
{actual}

Respond with JSON only: {{"score": <float>, "reason": "<one sentence>"}}
"""

_PLAN_PROMPT = """You are evaluating an agent's plan. Score plan quality
from 0.0 (no valid plan, plan contradicts task, or plan ignored) to 1.0
(clear ordered steps, all necessary, executed faithfully).

TASK:
{input}

PLAN STEPS:
{plan}

EXECUTED TOOLS:
{tools}

Respond with JSON only: {{"score": <float>, "reason": "<one sentence>"}}
"""


@register
class TaskCompletion(BaseMetric):
    """Judge: did the agent accomplish the task (trajectory + output)."""

    name = "task_completion"
    default_threshold = 0.7

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        judge = ctx.require_judge(self.name)
        prompt = _TASK_PROMPT.format(
            input=case.input,
            trajectory=_trajectory_summary(output.trajectory),
            actual=output.output or "",
        )
        verdict, usage = await judge.evaluate(prompt)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"judge_rationale": verdict.reason, **verdict.raw.get("details", {})},
            usage=usage,
        )


@register
class ReasoningCoherence(BaseMetric):
    """Judge: coherence of emitted thoughts (ReAct/CoT patterns).

    Skipped when the SUT emits no thought events — we never assume access
    to hidden chain-of-thought.
    """

    name = "reasoning_coherence"
    default_threshold = 0.6

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        trajectory = output.trajectory
        chain = trajectory.reasoning_chain if trajectory else []
        if not chain:
            return MetricResult.skipped(
                metric=self.name,
                reason="no thought events emitted — hidden CoT is not evaluated",
            )
        judge = ctx.require_judge(self.name)
        prompt = _REASONING_PROMPT.format(
            input=case.input, chain="\n".join(chain), actual=output.output or ""
        )
        verdict, usage = await judge.evaluate(prompt)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"judge_rationale": verdict.reason, "thought_count": len(chain)},
            usage=usage,
        )


@register
class PlanValidity(BaseMetric):
    """Judge: plan quality vs executed tools (Plan-and-Execute pattern)."""

    name = "plan_validity"
    default_threshold = 0.6

    async def measure(
        self, case: TestCase, output: TargetOutput, ctx: MetricContext
    ) -> MetricResult:
        trajectory = output.trajectory
        if trajectory is None:
            return MetricResult.skipped(metric=self.name, reason="no trajectory")
        plan_steps = [
            str(e.payload.get("text") or e.payload.get("step"))
            for e in trajectory.events
            if e.type in ("plan", "plan_step")
            or (e.type == "thought" and "plan" in str(e.payload.get("text", "")).lower())
        ]
        if not plan_steps:
            return MetricResult.skipped(
                metric=self.name, reason="no plan events in trajectory"
            )
        judge = ctx.require_judge(self.name)
        prompt = _PLAN_PROMPT.format(
            input=case.input,
            plan="\n".join(plan_steps),
            tools=", ".join(_names(_calls(output))) or "none",
        )
        verdict, usage = await judge.evaluate(prompt)
        return MetricResult(
            metric=self.name,
            status=MetricStatus.SCORED,
            score=verdict.score,
            passed=verdict.score >= self.threshold,
            reason=verdict.reason,
            details={"judge_rationale": verdict.reason, "plan_steps": plan_steps},
            usage=usage,
        )


def _trajectory_summary(trajectory: Trajectory | None, *, max_events: int = 40) -> str:
    """Compact text rendering of a trajectory for judge prompts."""
    if trajectory is None:
        return "(no trajectory)"
    lines = []
    for event in trajectory.events[:max_events]:
        agent = f"[{event.agent_id}] " if event.agent_id else ""
        payload = {k: v for k, v in event.payload.items() if v is not None}
        lines.append(f"{event.step_index}. {agent}{event.type}: {payload}")
    if len(trajectory.events) > max_events:
        lines.append(f"... ({len(trajectory.events) - max_events} more events)")
    return "\n".join(lines) or "(empty trajectory)"
