"""aitestkit — functional testing for agentic AI, RAG, and LLM apps."""

from aitestkit.assertions import assert_test
from aitestkit.models import (
    CaseResult,
    CaseStatus,
    ConversationCase,
    ConversationTurn,
    MetricResult,
    MetricStatus,
    RunMetadata,
    TargetOutput,
    TestCase,
    TestRun,
    ToolCall,
    TraceEvent,
    Trajectory,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "CaseResult",
    "CaseStatus",
    "ConversationCase",
    "ConversationTurn",
    "MetricResult",
    "MetricStatus",
    "RunMetadata",
    "TargetOutput",
    "TestCase",
    "TestRun",
    "ToolCall",
    "TraceEvent",
    "Trajectory",
    "Usage",
    "assert_test",
]
