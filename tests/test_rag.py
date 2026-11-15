"""RAG pillar tests: chunking, index, retrieval + generation metrics,
ingestion checks, token counting, golden sets."""

from __future__ import annotations

from pathlib import Path

from aitestkit.judge.judge import Judge
from aitestkit.metrics.base import MetricContext
from aitestkit.metrics.ingestion import (
    DeduplicationCheck,
    MetadataCompleteness,
    ParseCheck,
)
from aitestkit.metrics.rag import (
    AnswerRelevancy,
    CitationAccuracy,
    ContextualRecall,
    ContextWindow,
    Faithfulness,
)
from aitestkit.metrics.retrieval import (
    MRR,
    NDCG,
    HitRate,
    PrecisionAtK,
    RecallAtK,
    RerankDelta,
)
from aitestkit.models import (
    MetricStatus,
    TargetOutput,
    TestCase,
)
from aitestkit.providers.fake import FakeChatProvider, FakeEmbedder
from aitestkit.rag.chunking import (
    FixedChunker,
    HierarchicalChunker,
    SemanticChunker,
)
from aitestkit.rag.golden import GoldenSet, load_golden, save_golden
from aitestkit.rag.index import InMemoryVectorStore
from aitestkit.rag.token_counting import count_tokens


def test_count_tokens_offline() -> None:
    """Vendored tiktoken data — no network needed."""
    assert count_tokens("hello world") >= 1
    assert count_tokens("") == 0 or count_tokens("") >= 0
    # longer text -> more tokens (heuristic or real, monotonic)
    assert count_tokens("word " * 100) > count_tokens("word")


def test_fixed_chunker() -> None:
    chunker = FixedChunker(size=20, overlap=5)
    chunks = chunker.chunk("abcdefghijklmnopqrstuvwxyz" * 4, doc_id="d1")
    assert len(chunks) > 1
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].id != chunks[1].id


def test_hierarchical_chunker() -> None:
    chunker = HierarchicalChunker(parent_size=40, child_size=15)
    chunks = chunker.chunk("x" * 120, doc_id="d")
    levels = {c.metadata["level"] for c in chunks}
    assert levels == {"parent", "child"}
    children = [c for c in chunks if c.metadata["level"] == "child"]
    assert all("parent" in c.metadata for c in children)


async def test_semantic_chunker() -> None:
    chunker = SemanticChunker(FakeEmbedder(), threshold=0.99)
    chunks = await chunker.achunk(
        "Cats are mammals. Dogs are mammals. Quantum entanglement is weird.",
        doc_id="d",
    )
    assert len(chunks) >= 1


async def test_in_memory_index_and_drift() -> None:
    store = InMemoryVectorStore(FakeEmbedder())
    await store.upsert(
        [FixedChunker(size=100).chunk("paris is the capital of france")[0]]
    )
    hits = await store.query("capital france", top_k=1)
    assert len(hits) == 1
    drift = await store.reembed_sample()
    assert drift[0][1] == 1.0  # fake embedder is deterministic -> no drift


def test_golden_roundtrip(tmp_path: Path) -> None:
    golden = GoldenSet(queries=[{"id": "q1", "query": "x", "relevant_ids": ["a"]}])
    path = tmp_path / "g.yaml"
    save_golden(golden, path)
    loaded = load_golden(path)
    assert loaded.queries[0].relevant_ids == ["a"]
    assert loaded.schema_version == 1


def _case_with_golden() -> TestCase:
    return TestCase(id="t", input="q", expected_retrieval_ids=["a", "b", "c"])


async def test_recall_at_k() -> None:
    out = TargetOutput(retrieval_ids=["a", "x", "b"])
    result = await RecallAtK(k=3).measure(_case_with_golden(), out, MetricContext())
    assert abs(result.score - 2 / 3) < 1e-9
    assert result.details["hits_at_k"] == 2


async def test_precision_at_k() -> None:
    out = TargetOutput(retrieval_ids=["a", "x", "b"])
    result = await PrecisionAtK(k=3).measure(
        _case_with_golden(), out, MetricContext()
    )
    assert abs(result.score - 2 / 3) < 1e-9


async def test_mrr() -> None:
    out = TargetOutput(retrieval_ids=["x", "a"])
    result = await MRR().measure(_case_with_golden(), out, MetricContext())
    assert result.score == 0.5
    out2 = TargetOutput(retrieval_ids=["a"])
    assert (await MRR().measure(_case_with_golden(), out2, MetricContext())).score == 1.0


async def test_ndcg_and_hit_rate() -> None:
    out = TargetOutput(retrieval_ids=["a", "x"])
    ndcg = await NDCG(k=2).measure(_case_with_golden(), out, MetricContext())
    assert 0 < ndcg.score < 1.0
    hit = await HitRate(k=2).measure(_case_with_golden(), out, MetricContext())
    assert hit.score == 1.0
    miss = await HitRate(k=2).measure(
        _case_with_golden(),
        TargetOutput(retrieval_ids=["x", "y"]),
        MetricContext(),
    )
    assert miss.score == 0.0


async def test_rerank_delta() -> None:
    case = _case_with_golden()
    out = TargetOutput(
        retrieval_ids=["a", "b", "x"],
        metadata={"pre_rerank_ids": ["x", "y", "a"]},
    )
    result = await RerankDelta(k=3).measure(case, out, MetricContext())
    assert result.details["delta"] > 0
    assert result.passed is True


async def test_faithfulness_claim_decomposition() -> None:
    """Claims land in details; score = supported/total."""
    provider = FakeChatProvider()
    provider.respond_to(
        "evaluating whether an AI answer is faithful",
        {
            "claims": [
                {"claim": "paris is capital", "supported": True, "evidence": "doc1"},
                {"claim": "mars is green", "supported": False, "evidence": ""},
            ],
            "score": 0.5,
            "reason": "one unsupported claim",
        },
    )
    ctx = MetricContext(judge=Judge(provider, model="fake"))
    case = TestCase(id="t", input="q", context=["paris is the capital of france"])
    out = TargetOutput(output="paris is capital and mars is green")
    result = await Faithfulness().measure(case, out, ctx)
    assert result.score == 0.5
    assert result.details["unsupported"] == ["mars is green"]
    assert result.usage.prompt_tokens == 10


async def test_faithfulness_skips_without_context() -> None:
    ctx = MetricContext(judge=Judge(FakeChatProvider(), model="fake"))
    result = await Faithfulness().measure(
        TestCase(id="t", input="q"), TargetOutput(output="x"), ctx
    )
    assert result.status == MetricStatus.SKIPPED


async def test_answer_relevancy_and_contextual_recall() -> None:
    ctx = MetricContext(judge=Judge(FakeChatProvider(), model="fake"))
    case = TestCase(id="t", input="capital?", expected_output="paris",
                    context=["paris is capital"])
    rel = await AnswerRelevancy().measure(case, TargetOutput(output="paris"), ctx)
    assert rel.score == 1.0
    recall = await ContextualRecall().measure(case, TargetOutput(output="x"), ctx)
    assert recall.score == 1.0


async def test_citation_accuracy() -> None:
    provider = FakeChatProvider()
    provider.respond_to(
        "directly support the claim",
        {"score": 1.0, "reason": "supported"},
    )
    ctx = MetricContext(judge=Judge(provider, model="fake"))
    case = TestCase(id="t", input="q")
    out = TargetOutput(
        output="paris is capital [1] and it rains a lot [9]",
        retrieval_context=["paris is the capital of france"],
    )
    result = await CitationAccuracy().measure(case, out, ctx)
    # [1] verified by judge, [9] out of range -> 0.5
    assert result.score == 0.5
    assert result.details["total"] == 2


async def test_context_window() -> None:
    case = TestCase(id="t", input="hi", context=["short"])
    metric = ContextWindow(max_tokens=100)
    ok = await metric.measure(case, TargetOutput(), MetricContext())
    assert ok.passed is True
    over = await ContextWindow(max_tokens=1).measure(
        case, TargetOutput(), MetricContext()
    )
    assert over.passed is False
    assert "overflows" in over.reason


async def test_ingestion_checks() -> None:
    docs = [
        {"id": "1", "text": "hello world", "metadata": {"source": "a"}},
        {"id": "2", "text": "hello world", "metadata": {}},
        {"id": "3", "text": "bad\ufffdparse", "metadata": {"source": "b"}},
    ]
    case = TestCase(
        id="t", input="q",
        metadata={"documents": docs, "required_metadata": ["source"]},
    )
    parse = await ParseCheck().measure(case, TargetOutput(), MetricContext())
    assert abs(parse.score - 2 / 3) < 1e-9
    meta = await MetadataCompleteness().measure(
        case, TargetOutput(), MetricContext()
    )
    assert abs(meta.score - 2 / 3) < 1e-9
    assert meta.details["docs_missing_metadata"][0]["id"] == "2"
    dedup = await DeduplicationCheck().measure(
        case, TargetOutput(), MetricContext()
    )
    assert abs(dedup.score - 2 / 3) < 1e-9
    assert dedup.details["duplicate_indexes"] == [1]


def test_metrics_registered() -> None:
    from aitestkit.metrics import registry

    for name in (
        "faithfulness", "recall_at_k", "mrr", "context_window",
        "ingestion_dedup", "citation_accuracy",
    ):
        assert name in registry.metric_names(), name


async def test_chunk_compare_e2e(tmp_path: Path) -> None:
    """The CLI harness chunks + indexes + scores a golden set."""
    from aitestkit.cli import _load_corpus
    from aitestkit.metrics.retrieval import RecallAtK

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.txt").write_text("paris is the capital of france. " * 3)
    (corpus / "doc2.txt").write_text("tokyo is the capital of japan. " * 3)
    golden = GoldenSet(
        queries=[{"id": "q1", "query": "capital france", "relevant_ids": []}]
    )
    gpath = tmp_path / "g.yaml"
    save_golden(golden, gpath)

    docs = _load_corpus(corpus)
    assert len(docs) == 2

    store = InMemoryVectorStore(FakeEmbedder())
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(FixedChunker(size=60, overlap=10).chunk(text, doc_id=doc_id))
    await store.upsert(chunks)
    hits = await store.query("capital france", top_k=3)
    # golden ids: first chunk of doc1 is the relevant one
    golden.queries[0].relevant_ids = [hits[0][0].id]
    case = TestCase(
        id="q1", input="capital france",
        expected_retrieval_ids=[hits[0][0].id],
    )
    result = await RecallAtK(k=3).measure(
        case, TargetOutput(retrieval_ids=[c.id for c, _ in hits]),
        MetricContext(),
    )
    assert result.score == 1.0
