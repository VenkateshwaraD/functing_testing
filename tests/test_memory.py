"""Memory pillar tests: stores, conversation runner, memory metrics."""

from __future__ import annotations

from pathlib import Path

from aitestkit.agentic.memory import (
    DictMemoryStore,
    JsonFileMemoryStore,
)
from aitestkit.metrics.base import MetricContext
from aitestkit.metrics.memory import (
    ContextRetention,
    ContradictionResolution,
    ForgettingCheck,
    MemoryAccuracy,
)
from aitestkit.models import (
    CaseStatus,
    MetricStatus,
    TargetOutput,
    TestCase,
)


def test_dict_store_sessions_isolated() -> None:
    store = DictMemoryStore()
    store.set("k", "v1", session_id="s1")
    store.set("k", "v2", session_id="s2")
    assert store.get("k", session_id="s1") == "v1"
    assert store.get("k", session_id="s2") == "v2"
    store.clear(session_id="s1")
    assert store.get("k", session_id="s1") is None
    assert store.get("k", session_id="s2") == "v2"


def test_json_store_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    store = JsonFileMemoryStore(path)
    store.set("fact", "paris", session_id="s")
    # new instance = simulated process restart
    reloaded = JsonFileMemoryStore(path)
    assert reloaded.get("fact", session_id="s") == "paris"


def test_search_finds_values() -> None:
    store = DictMemoryStore()
    store.set("city", "paris", session_id="s")
    assert store.search("paris", session_id="s") == ["paris"]
    assert store.search("london", session_id="s") == []


async def test_context_retention() -> None:
    case = TestCase(
        id="t", input="what's my name",
        metadata={"must_remember": ["ada"]},
    )
    output = TargetOutput(output="your name is ada")
    result = await ContextRetention().measure(case, output, MetricContext())
    assert result.passed is True
    lost = await ContextRetention().measure(
        case, TargetOutput(output="i don't know"), MetricContext()
    )
    assert lost.score == 0.0
    assert lost.details["lost"] == ["ada"]


async def test_contradiction_resolution() -> None:
    case = TestCase(
        id="t", input="where do i live",
        metadata={"superseded": "london", "current": "paris"},
    )
    good = await ContradictionResolution().measure(
        case, TargetOutput(output="you live in paris"), MetricContext()
    )
    assert good.score == 1.0
    stale = await ContradictionResolution().measure(
        case, TargetOutput(output="you live in london"), MetricContext()
    )
    assert stale.score == 0.0
    mixed = await ContradictionResolution().measure(
        case,
        TargetOutput(output="you lived in london, now paris"),
        MetricContext(),
    )
    assert mixed.score == 0.5


async def test_forgetting_check_direct_inspection() -> None:
    store = DictMemoryStore()
    store.set("user_name", "ada", session_id="s1")
    ctx = MetricContext(memory_store=store)
    case = TestCase(
        id="t", input="q",
        metadata={"memory_key": "user_name", "session_id": "s1"},
    )
    result = await ForgettingCheck().measure(case, TargetOutput(), ctx)
    assert result.passed is True
    missing = TestCase(
        id="t2", input="q", metadata={"memory_key": "gone", "session_id": "s1"}
    )
    result2 = await ForgettingCheck().measure(missing, TargetOutput(), ctx)
    assert result2.score == 0.0
    no_store = await ForgettingCheck().measure(
        case, TargetOutput(), MetricContext()
    )
    assert no_store.status == MetricStatus.ERROR


async def test_memory_accuracy_judge() -> None:
    from aitestkit.judge.judge import Judge
    from aitestkit.providers.fake import FakeChatProvider

    store = DictMemoryStore()
    store.set("name", "ada", session_id="s")
    ctx = MetricContext(
        judge=Judge(FakeChatProvider(), model="fake"), memory_store=store
    )
    case = TestCase(id="t", input="q", metadata={"session_id": "s"})
    result = await MemoryAccuracy().measure(
        case, TargetOutput(output="your name is ada"), ctx
    )
    assert result.status == MetricStatus.SCORED
    empty = DictMemoryStore()
    ctx2 = MetricContext(
        judge=Judge(FakeChatProvider(), model="fake"), memory_store=empty
    )
    skipped = await MemoryAccuracy().measure(
        TestCase(id="t", input="q"), TargetOutput(output="x"), ctx2
    )
    assert skipped.status == MetricStatus.SKIPPED


async def test_conversation_runner_sessions() -> None:
    """Multi-turn conversation shares session_id; memory persists per session."""
    import memory_app

    from aitestkit.metrics.deterministic import Contains
    from aitestkit.models import ConversationCase, ConversationTurn
    from aitestkit.runner import RunEngine
    from aitestkit.targets.python_target import PythonTarget

    memory_app.store.clear()
    engine = RunEngine(
        target=PythonTarget(spec="memory_app:chat"),
        metrics=[Contains(substrings=["ada"], any_of=True)],
        ctx=MetricContext(memory_store=memory_app.store),
        conversations=[
            ConversationCase(
                id="conv1",
                session_id="sess-1",
                turns=[
                    ConversationTurn(input="my name is ada"),
                    ConversationTurn(input="what is my name", expected_output=None),
                ],
            )
        ],
    )
    test_run, _ = await engine.run([])
    assert len(test_run.results) == 2
    assert all(r.session_id == "sess-1" for r in test_run.results)
    # turn-1 recall works because session memory persisted
    assert test_run.results[1].actual_output == "your name is ada"
    assert test_run.results[1].status == CaseStatus.PASSED
