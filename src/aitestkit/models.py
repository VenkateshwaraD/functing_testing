"""Core data models: test cases, results, statuses, usage, run metadata.

This is the versioned result schema every pillar reports into. ``SCHEMA_VERSION``
bumps whenever the serialized shape changes; loaders run migrations.
"""

from __future__ import annotations

import enum
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class CaseStatus(enum.StrEnum):
    """Outcome of a test case. ``error`` means infrastructure, not quality."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    FLAKY = "flaky"
    XFAIL = "xfailed"  # expected failure — detection confirmed (expect: fail)


class MetricStatus(enum.StrEnum):
    """Outcome of a single metric evaluation."""

    SCORED = "scored"
    ERROR = "error"
    SKIPPED = "skipped"


class Usage(BaseModel):
    """Token usage and cost attributed to one call or aggregation scope."""

    model_config = ConfigDict(frozen=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    provider: str | None = None
    model: str | None = None

    def __iadd__(self, other: Usage) -> Usage:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        if self.provider is None:
            self.provider = other.provider
            self.model = other.model
        return self


class ToolCall(BaseModel):
    """One tool invocation observed (or expected) in an agent run."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    requires_confirmation: bool = False


class TraceEvent(BaseModel):
    """One step in an agent trajectory."""

    type: str  # thought|action|observation|tool_call|llm_call|message|handoff|memory_op|retrieval
    agent_id: str | None = None
    step_index: int = 0
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Ordered trace of an agent run; serializes with a schema version."""

    schema_version: int = SCHEMA_VERSION
    events: list[TraceEvent] = Field(default_factory=list)
    finish_reason: str | None = None  # done|max_steps|timeout|error|cancelled

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls = []
        for event in self.events:
            if event.type != "tool_call":
                continue
            calls.append(
                ToolCall(
                    name=str(event.payload.get("name", "")),
                    args=dict(event.payload.get("args", {})),
                    result=event.payload.get("result"),
                    error=event.payload.get("error"),
                )
            )
        return calls

    @property
    def reasoning_chain(self) -> list[str]:
        return [str(e.payload.get("text", "")) for e in self.events if e.type == "thought"]

    @property
    def agent_sequence(self) -> list[str]:
        return [
            str(e.payload.get("to_agent", ""))
            for e in self.events
            if e.type == "handoff" and e.payload.get("to_agent")
        ]

    @property
    def step_signatures(self) -> list[str]:
        """Hashable per-step signature used for loop detection and replay diffs."""
        signatures = []
        for event in self.events:
            key = event.payload.get("name") or event.payload.get("text") or ""
            signatures.append(f"{event.type}:{event.agent_id}:{key}")
        return signatures


class TestCase(BaseModel):
    """One functional test case against the system under test."""

    id: str | None = None
    input: str
    expected_output: str | None = None
    context: list[str] | None = None  # context the case supplies (e.g. seeded docs)
    retrieval_context: list[str] | None = None  # captured retrieved chunks
    expected_retrieval_ids: list[str] | None = None  # golden labelled ids
    expected_tools: list[ToolCall | str] | None = None
    expected_trajectory: Trajectory | None = None
    tags: list[str] = Field(default_factory=list)  # incl. capability tags for coverage
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Expected outcome — `fail` marks a detection probe (strict xfail):
    # a scored metric failing → xfailed; everything passing → failed.
    expect: Literal["fail"] | None = None
    # Per-case metric selection — overrides the suite's metrics: list.
    # YAML `assert:` accepts metric names or {name: ..., params} dicts.
    assert_: list[Any] | None = Field(default=None, alias="assert")
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ConversationTurn(BaseModel):
    """One turn in a multi-turn conversational test."""

    model_config = ConfigDict(extra="forbid")

    input: str
    expected_output: str | None = None
    tags: list[str] = Field(default_factory=list)


class ConversationCase(BaseModel):
    """A multi-turn test case (conversational RAG, memory probes)."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    turns: list[ConversationTurn]
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetOutput(BaseModel):
    """What a target adapter returns for one invocation."""

    output: str | None = None
    trajectory: Trajectory | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    retrieval_context: list[str] | None = None
    retrieval_ids: list[str] | None = None  # ids of retrieved chunks, ranked
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # stage info, e.g. pre_rerank_ids


class MetricResult(BaseModel):
    """Outcome of evaluating one metric on one case."""

    metric: str
    status: MetricStatus = MetricStatus.SCORED
    score: float | None = None
    passed: bool | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    duration_ms: float = 0.0

    @classmethod
    def error(cls, *, metric: str, reason: str) -> MetricResult:
        return cls(metric=metric, status=MetricStatus.ERROR, passed=False, reason=reason)

    @classmethod
    def skipped(cls, *, metric: str, reason: str) -> MetricResult:
        return cls(metric=metric, status=MetricStatus.SKIPPED, reason=reason)


class CaseResult(BaseModel):
    """Everything known about one case after a run."""

    case_id: str
    status: CaseStatus
    input: str
    actual_output: str | None = None
    error: str | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    trajectory: Trajectory | None = None
    latency_ms: float = 0.0
    judge_usage: Usage = Field(default_factory=Usage)
    sut_usage: Usage = Field(default_factory=Usage)
    attribution: str | None = None  # e.g. "retrieval" | "generation" | "rerank"
    tags: list[str] = Field(default_factory=list)
    repeat: dict[str, Any] = Field(default_factory=dict)  # pass_rate, runs, outcomes
    session_id: str | None = None  # conversation/memory session
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Reproducibility record attached to every TestRun."""

    git_sha: str | None = None
    aitestkit_version: str = ""
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    judge: str | None = None
    embedder: str | None = None
    region: str | None = None  # execution region, e.g. REGION env (qa/uat)
    model_versions: dict[str, str] = Field(default_factory=dict)  # prompt/kb/chunker/embedder
    seed: int | None = None
    config_hash: str | None = None
    python_version: str = ""
    platform: str = ""
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None


class GateResult(BaseModel):
    """Evaluation of one quality gate."""

    name: str
    passed: bool
    actual: float | int | None = None
    expected: str = ""
    detail: str = ""


class TestRun(BaseModel):
    """A complete suite run — the top-level serialized artifact."""

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    suite_name: str = ""
    metadata: RunMetadata = Field(default_factory=RunMetadata)
    results: list[CaseResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    gates: list[GateResult] = Field(default_factory=list)
    baseline_tag: str | None = None


# These are domain models, not pytest test classes — stop pytest collection.
TestCase.__test__ = False  # type: ignore[attr-defined]
TestRun.__test__ = False  # type: ignore[attr-defined]
