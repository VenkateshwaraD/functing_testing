"""Vector DB pillar tests: factory, checks metrics, lazy import errors."""

from __future__ import annotations

import pytest

from aitestkit.exceptions import ConfigError
from aitestkit.metrics.base import MetricContext
from aitestkit.models import MetricStatus, TargetOutput, TestCase
from aitestkit.providers.fake import FakeEmbedder
from aitestkit.rag.chunking import Chunk
from aitestkit.vectordb.base import build_vector_store
from aitestkit.vectordb.checks import EmbeddingDrift, VectorDbCount


def test_factory_memory_store() -> None:
    store = build_vector_store({"type": "memory"}, embedder=FakeEmbedder())
    assert store is not None


def test_factory_unknown_type() -> None:
    with pytest.raises(ConfigError, match="unknown vectordb"):
        build_vector_store({"type": "bogus"}, embedder=FakeEmbedder())


def test_factory_chroma_missing_extra() -> None:
    try:
        import chromadb  # noqa: F401

        pytest.skip("chromadb installed")
    except ImportError:
        pass
    with pytest.raises(ConfigError, match="aitestkit\\[chroma\\]"):
        build_vector_store({"type": "chroma"}, embedder=FakeEmbedder())


def test_factory_faiss_missing_extra() -> None:
    try:
        import faiss  # noqa: F401

        pytest.skip("faiss installed")
    except ImportError:
        pass
    with pytest.raises(ConfigError, match="aitestkit\\[faiss\\]"):
        build_vector_store({"type": "faiss"}, embedder=FakeEmbedder())


async def test_count_check() -> None:
    store = build_vector_store({"type": "memory"}, embedder=FakeEmbedder())
    await store.upsert(
        [Chunk(id="c1", text="one"), Chunk(id="c2", text="two")]
    )
    ctx = MetricContext(vector_store=store)
    ok = await VectorDbCount(min_count=2).measure(
        TestCase(id="t", input="q"), TargetOutput(), ctx
    )
    assert ok.passed is True
    exact = await VectorDbCount(expected=3).measure(
        TestCase(id="t", input="q"), TargetOutput(), ctx
    )
    assert exact.passed is False
    assert "expected 3, got 2" in exact.reason


async def test_count_check_no_store() -> None:
    result = await VectorDbCount().measure(
        TestCase(id="t", input="q"), TargetOutput(), MetricContext()
    )
    assert result.status == MetricStatus.ERROR


async def test_embedding_drift_no_drift() -> None:
    store = build_vector_store({"type": "memory"}, embedder=FakeEmbedder())
    await store.upsert([Chunk(id="c1", text="stable text")])
    ctx = MetricContext(vector_store=store)
    result = await EmbeddingDrift().measure(
        TestCase(id="t", input="q"), TargetOutput(), ctx
    )
    assert result.score == 1.0
    assert result.passed is True


async def test_embedding_drift_detects_change() -> None:
    """A changed embedder drops stored-vs-fresh similarity."""
    store = build_vector_store({"type": "memory"}, embedder=FakeEmbedder())
    await store.upsert([Chunk(id="c1", text="stored under old model")])
    # swap embedder for a different deterministic but different one
    store.embedder = FakeEmbedder(dims=8)
    # fake embedder is hash-based -> same vectors; simulate real drift by
    # corrupting the stored vector
    store._vectors[0] = [1.0] + [0.0] * 7
    ctx = MetricContext(vector_store=store)
    result = await EmbeddingDrift(threshold=0.99).measure(
        TestCase(id="t", input="q"), TargetOutput(), ctx
    )
    assert result.score < 1.0
    assert result.details["worst_chunk"] == "c1"


async def test_embedding_drift_empty_store() -> None:
    store = build_vector_store({"type": "memory"}, embedder=FakeEmbedder())
    ctx = MetricContext(vector_store=store)
    result = await EmbeddingDrift().measure(
        TestCase(id="t", input="q"), TargetOutput(), ctx
    )
    assert result.status == MetricStatus.SKIPPED


async def test_memory_store_query_scores() -> None:
    """InMemoryVectorStore conforms to the shared protocol end to end."""
    store = build_vector_store({"type": "memory"}, embedder=FakeEmbedder())
    await store.upsert(
        [Chunk(id="a", text="paris capital"), Chunk(id="b", text="tokyo")]
    )
    hits = await store.query("paris", top_k=2)
    assert len(hits) == 2
    assert {c.id for c, _ in hits} == {"a", "b"}
    assert await store.count() == 2
